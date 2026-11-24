"""Local MLP-block residual recomputation ("replay") for dense Gemma-4 decoder layers.

Used to measure the residual energy injected by language-neuron amplification (scaling selected
down_proj input coordinates) so random controls can be energy-matched. Fails closed for unsupported
block recipes (MoE, per-layer inputs, missing components).
"""
from contextlib import contextmanager
import torch


def validate_layers(layers):
    for layer in layers:
        if type(layer).__name__ != 'Gemma4TextDecoderLayer':
            raise ValueError('This runner supports dense Gemma4TextDecoderLayer only')
        if layer.enable_moe_block or layer.hidden_size_per_layer_input:
            raise ValueError('MoE/per-layer input requires a separate replay adapter')
        for name in ('mlp','post_feedforward_layernorm','pre_feedforward_layernorm','layer_scalar'):
            if not hasattr(layer,name): raise ValueError(f'Missing residual component: {name}')


def tail(layer, activation, residual):
    """Matches the validated dense Gemma4 block tail including bf16 residual addition."""
    y=layer.mlp.down_proj(activation)
    return (residual + layer.post_feedforward_layernorm(y)) * layer.layer_scalar


def amplified(activation, indices, increment):
    a=activation.clone()
    a[...,indices] *= 1.+increment
    return a


def energy(output, reference, lengths):
    d=(output.float()-reference.float()).square().sum(-1)
    chunks=d.split(lengths,dim=1)
    return dict(mean=sum(float(c.mean()) for c in chunks)/len(chunks),
                last=sum(float(c[:,-1].mean()) for c in chunks)/len(chunks))


@torch.inference_mode()
def capture(model, layers, tokens, active):
    """Capture unmodified S=U states; no caching and no intervention hooks."""
    rows={}; handles=[]
    for l in active:
        rows[l]={}
        def residual_hook(m,a,l=l): rows[l]['residual']=a[0].detach().cpu().clone()
        def activation_hook(m,a,l=l): rows[l]['activation']=a[0].detach().cpu().clone()
        def output_hook(m,a,o,l=l): rows[l]['output']=o.detach().cpu().clone()
        handles.extend([
            layers[l].pre_feedforward_layernorm.register_forward_pre_hook(residual_hook),
            layers[l].mlp.down_proj.register_forward_pre_hook(activation_hook),
            layers[l].register_forward_hook(output_hook)])
    try: model(input_ids=tokens,use_cache=False,logits_to_keep=1)
    finally:
        for h in handles:h.remove()
    return rows


class LayerReplay:
    def __init__(self, layer, rows):
        self.layer=layer
        device=layer.mlp.down_proj.weight.device
        self.lengths=[r['activation'].shape[1] for r in rows]
        self.a=torch.cat([r['activation'] for r in rows],dim=1).to(device)
        self.res=torch.cat([r['residual'] for r in rows],dim=1).to(device)
        self.base=tail(layer,self.a,self.res)
        observed=torch.cat([r['output'] for r in rows],dim=1).to(device)
        diff=(self.base.float()-observed.float()).square().mean().sqrt()
        denom=observed.float().square().mean().sqrt().clamp_min(1e-12)
        self.replay_relative_rms=float(diff/denom)
        # Concatenated GEMM can differ slightly in bf16; record, never silently ignore.
        if not torch.isfinite(diff) or self.replay_relative_rms > .005:
            raise ValueError(f'Block replay mismatch {self.replay_relative_rms}')
        self.residual_energy=float(observed.float().square().sum(-1).mean())

    def measure(self,indices,increment):
        out=tail(self.layer,amplified(self.a,indices,increment),self.res)
        result=energy(out,self.base,self.lengths)
        if not all(torch.isfinite(torch.tensor(v)) for v in result.values()):
            raise ValueError('Nonfinite residual energy')
        return result


@contextmanager
def intervention(layers, groups, increments, telemetry=False):
    """Apply language-neuron amplification (coordinate multiplication of down_proj inputs).

    With telemetry=True, each block's unamplified output is recomputed from the same state after the
    real block finishes; it never substitutes model output. Telemetry measures the local direct change
    on each arm's own trajectory. For Gemma base models the suppression arm equals untreated (S=U).
    """
    handles=[]; cache={}; traces=[]
    for l,indices in groups.items():
        layer=layers[l]; increment=increments[l]
        def pre(m,a,l=l,indices=indices,increment=increment):
            if telemetry: cache.setdefault(l,{})['activation']=a[0].detach().clone()
            return (amplified(a[0],indices,increment),)+tuple(a[1:])
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
        if telemetry:
            def res_hook(m,a,l=l): cache.setdefault(l,{})['residual']=a[0].detach().clone()
            def out_hook(m,a,o,l=l,layer=layer):
                state=cache[l]
                # Calling down_proj directly would re-enter the intervention prehook.
                # Temporarily suppress it via the explicit functional linear below.
                activation=state['activation']
                linear=layer.mlp.down_proj
                y=torch.nn.functional.linear(activation,linear.weight,linear.bias)
                baseline=(state['residual']+layer.post_feedforward_layernorm(y))*layer.layer_scalar
                d=(o.float()-baseline.float()).square().sum(-1)
                traces.append(dict(layer=l,tokens=o.shape[1],energy=float(d.mean()),last_energy=float(d[:,-1].mean()),
                                   baseline_energy=float(baseline.float().square().sum(-1).mean())))
                cache.pop(l,None)
            handles.extend([layer.pre_feedforward_layernorm.register_forward_pre_hook(res_hook),
                            layer.register_forward_hook(out_hook)])
    try: yield traces
    finally:
        for h in handles:h.remove()


def synthetic_test():
    """Detect missing postnorm/scalar, wrong index scaling, and intervention hook leakage."""
    from types import SimpleNamespace
    class Norm(torch.nn.Module):
        def forward(self,x):return x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_feedforward_layernorm=torch.nn.Identity()
            self.mlp=SimpleNamespace(down_proj=torch.nn.Linear(8,8,bias=False))
            self.post_feedforward_layernorm=Norm()
            self.layer_scalar=torch.tensor([.7])
        def forward(self,x):
            r=self.pre_feedforward_layernorm(x)
            return (r+self.post_feedforward_layernorm(self.mlp.down_proj(x)))*self.layer_scalar
    torch.manual_seed(1);layer=Block();x=torch.randn(1,5,8);base=layer(x)
    assert torch.equal(base,tail(layer,x,x))
    rows=[dict(activation=x,residual=x,output=base)]
    replay=LayerReplay(layer,rows)
    assert replay.measure([1,3],0)['mean']==0
    expected=tail(layer,amplified(x,[1,3],2),x)
    with intervention([layer],{0:[1,3]},{0:2},telemetry=True) as trace:
        actual=layer(x)
    assert torch.equal(actual,expected)
    assert abs(trace[0]['energy']-replay.measure([1,3],2)['mean'])<1e-6
    assert torch.equal(layer(x),base)
    assert not layer.mlp.down_proj._forward_pre_hooks
    print('[test] synthetic postnorm/scalar replay, zero dose, original multiplication, telemetry and cleanup PASS',flush=True)
