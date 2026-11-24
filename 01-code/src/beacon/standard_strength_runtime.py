"""Language-neuron amplification and local MLP-tail residual recomputation ("replay") for dense
Llama, Qwen3.5, and Gemma-4 decoder layers, with optional frozen suppression of Spanish neurons.

The replay recomputes a block's MLP tail from captured activations so the residual energy injected by
amplifying selected down_proj input coordinates can be measured and matched by random controls.
"""
from contextlib import contextmanager
import torch

GEMMA='Gemma4TextDecoderLayer'


def tensor_output(output):
    return output[0] if isinstance(output, tuple) else output


def is_gemma(layer):
    return type(layer).__name__==GEMMA


def validate_layers(layers):
    for layer in layers:
        name=type(layer).__name__
        if name==GEMMA:
            if getattr(layer,'enable_moe_block',False) or getattr(layer,'hidden_size_per_layer_input',0):
                raise ValueError('MoE/per-layer input requires a separate replay adapter')
            for part in ('pre_feedforward_layernorm','post_feedforward_layernorm','layer_scalar'):
                if not hasattr(layer,part):raise ValueError(f'Missing Gemma residual component: {part}')
            if not hasattr(layer.mlp,'down_proj'):raise ValueError('Expected dense Gemma MLP tail')
            continue
        if name not in ('LlamaDecoderLayer','Qwen3_5DecoderLayer'):
            raise ValueError(f'Unsupported block: {name}')
        if not hasattr(layer,'post_attention_layernorm') or not hasattr(layer.mlp,'down_proj'):
            raise ValueError('Expected dense standard MLP tail')
        if hasattr(layer,'post_feedforward_layernorm') or hasattr(layer,'layer_scalar'):
            raise ValueError('Unexpected postnorm/scalar recipe')


def residual_site(layer):
    """Module whose input is the residual stream entering the MLP tail."""
    return layer.pre_feedforward_layernorm if is_gemma(layer) else layer.post_attention_layernorm


def tail(layer, activation, residual):
    p=layer.mlp.down_proj
    y=torch.nn.functional.linear(activation,p.weight,p.bias)
    if is_gemma(layer):
        # Validated dense Gemma-4 tail, including BF16 residual addition and the layer scalar.
        return (residual+layer.post_feedforward_layernorm(y))*layer.layer_scalar
    return residual+y


@contextmanager
def intervention(layers,groups,increments,suppression=None,factor=1.,telemetry=False):
    """Amplify selected coordinates, then apply suppression S, in a fixed order so BF16 rounding on
    overlapping coordinates is deterministic. With telemetry=True, records per-layer injected energy."""
    suppression=suppression or {};handles=[];cache={};traces=[]
    active=set(groups)| (set(suppression) if factor!=1 else set())
    try:
        for l in sorted(active):
            layer=layers[l];ix=groups.get(l,[]);sx=suppression.get(l,[])
            increment=increments.get(l,0.)
            def pre(m,args,l=l,ix=ix,sx=sx,increment=increment):
                a=args[0].clone()
                if telemetry and l in groups:
                    baseline=a.clone()
                    if factor!=1:baseline[...,sx]*=factor
                    cache.setdefault(l,{})['activation']=baseline
                a[...,ix]*=1+increment
                if factor!=1:a[...,sx]*=factor
                return (a,)+tuple(args[1:])
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
            if telemetry and l in groups:
                def residual(m,args,l=l):cache.setdefault(l,{})['residual']=args[0].detach().clone()
                def output(m,args,out,l=l,layer=layer):
                    state=cache.pop(l);base=tail(layer,state['activation'],state['residual']);actual=tensor_output(out)
                    d=(actual.float()-base.float()).square().sum(-1)
                    traces.append(dict(layer=l,tokens=actual.shape[1],energy=float(d.mean()),
                                       last_energy=float(d[:,-1].mean()),baseline_energy=float(base.float().square().sum(-1).mean())))
                handles.extend([residual_site(layer).register_forward_pre_hook(residual),
                                layer.register_forward_hook(output)])
        yield traces
    finally:
        for h in handles:h.remove()


@torch.inference_mode()
def capture(model,layers,tokens,active,suppression,factor):
    rows={l:{} for l in active};handles=[]
    try:
            for l in active:
                def residual(m,args,l=l):rows[l]['residual']=args[0].detach().cpu().clone()
                def activation(m,args,l=l):rows[l]['activation']=args[0].detach().cpu().clone()
                def output(m,args,out,l=l):rows[l]['output']=tensor_output(out).detach().cpu().clone()
                handles.extend([residual_site(layers[l]).register_forward_pre_hook(residual),
                    layers[l].mlp.down_proj.register_forward_pre_hook(activation),layers[l].register_forward_hook(output)])
            with intervention(layers,{}, {},suppression,factor):
                model(input_ids=tokens,use_cache=False,logits_to_keep=1)
    finally:
        for h in handles:h.remove()
    if any(set(r)!={'residual','activation','output'} for r in rows.values()):raise ValueError('Incomplete capture')
    return rows


class LayerReplay:
    """Local replay of one block tail.

    ``positions`` optionally gives, per row, the prompt positions averaged for ``mean``
    (e.g. the user-message span under a chat template); ``all`` always averages every
    position and ``last`` is the final (generation) position.
    """
    @torch.inference_mode()
    def __init__(self,layer,rows,suppression_indices=(),factor=1.,positions=None):
        self.layer=layer;device=layer.mlp.down_proj.weight.device
        self.suppression_indices=list(suppression_indices);self.factor=factor
        self.rows=[{k:v.to(device) for k,v in r.items()} for r in rows]
        if positions is not None:
            if len(positions)!=len(self.rows) or any(not p for p in positions):raise ValueError('Positions per row required')
            for r,p in zip(self.rows,positions):
                if max(p)>=r['output'].shape[1] or min(p)<0:raise ValueError('Position outside prompt')
        self.positions=[list(p) for p in positions] if positions is not None else None
        errors=[]
        for r in self.rows:
            a=r['activation'].clone()
            if factor!=1:a[...,self.suppression_indices]*=factor
            base=tail(layer,a,r['residual']);r['base']=base
            error=float((base.float()-r['output'].float()).square().mean().sqrt()/r['output'].float().square().mean().sqrt().clamp_min(1e-12))
            if not torch.isfinite(base).all() or error>.005:raise ValueError(f'Replay mismatch {error}')
            errors.append(error)
        self.replay_relative_rms=max(errors)
        self.residual_energy=sum(float(r['output'].float().square().sum(-1).mean()) for r in self.rows)/len(self.rows)

    @torch.inference_mode()
    def measure(self,indices,increment):
        means=[];alls=[];lasts=[]
        for i,r in enumerate(self.rows):
            a=r['activation'].clone();a[...,indices]*=1+increment
            if self.factor!=1:a[...,self.suppression_indices]*=self.factor
            d=(tail(self.layer,a,r['residual']).float()-r['base'].float()).square().sum(-1)
            if not torch.isfinite(d).all():raise ValueError('Nonfinite energy')
            alls.append(float(d.mean()));lasts.append(float(d[:,-1].mean()))
            means.append(float(d[:,self.positions[i]].mean()) if self.positions is not None else alls[-1])
        n=len(self.rows)
        return dict(mean=sum(means)/n,all=sum(alls)/n,last=sum(lasts)/n)


@torch.inference_mode()
def synthetic_test():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__();self.post_attention_layernorm=torch.nn.Identity()
            self.mlp=torch.nn.Module();self.mlp.down_proj=torch.nn.Linear(8,8)
        def forward(self,x):return x+self.mlp.down_proj(self.post_attention_layernorm(x))
    torch.manual_seed(11);layer=Block();x=torch.randn(1,5,8);original=layer(x)
    s=x.clone();s[...,[1,2]]*=.1
    a=x.clone();a[...,[2,3]]*=5;a[...,[1,2]]*=.1
    with intervention([layer],{0:[2,3]},{0:4},{0:[1,2]},.1,True) as trace:actual=layer(x)
    expected=tail(layer,a,x);baseline=tail(layer,s,x)
    assert torch.equal(actual,expected)
    replay=LayerReplay(layer,[dict(activation=x,residual=x,output=baseline)],[1,2],.1)
    assert replay.measure([2,3],0)['mean']==0
    assert abs(trace[0]['energy']-replay.measure([2,3],4)['mean'])<1e-6
    assert torch.equal(layer(x),original)
    assert not layer.mlp.down_proj._forward_pre_hooks
    try:
        with intervention([layer],{0:[2]},{0:1},{0:[1]},.1):raise RuntimeError('test cleanup')
    except RuntimeError:pass
    assert not layer.mlp.down_proj._forward_pre_hooks
    gemma=synthetic_gemma_block();y=torch.randn(1,5,8);g_original=gemma(y)
    gs=y.clone();gs[...,[1,2]]*=.1
    ga=y.clone();ga[...,[2,3]]*=5;ga[...,[1,2]]*=.1
    with intervention([gemma],{0:[2,3]},{0:4},{0:[1,2]},.1,True) as g_trace:g_actual=gemma(y)
    assert torch.allclose(g_actual,tail(gemma,ga,y))
    g_replay=LayerReplay(gemma,[dict(activation=y,residual=y,output=tail(gemma,gs,y))],[1,2],.1)
    assert g_replay.measure([2,3],0)['mean']==0
    assert abs(g_trace[0]['energy']-g_replay.measure([2,3],4)['mean'])<1e-5
    assert torch.equal(gemma(y),g_original)
    assert not gemma.mlp.down_proj._forward_pre_hooks and not gemma.pre_feedforward_layernorm._forward_pre_hooks
    print('[test] standard residual replay, overlapping suppression, telemetry and cleanup PASS (dense and Gemma-4 tails)',flush=True)


def synthetic_gemma_block():
    """Minimal block with the Gemma-4 MLP tail recipe; residual passes through pre_feedforward_layernorm."""
    def init(self):
        torch.nn.Module.__init__(self)
        self.pre_feedforward_layernorm=torch.nn.Identity()
        self.post_feedforward_layernorm=torch.nn.LayerNorm(8)
        self.mlp=torch.nn.Module();self.mlp.down_proj=torch.nn.Linear(8,8)
        self.layer_scalar=torch.tensor(.5)
    def forward(self,x):
        return (x+self.post_feedforward_layernorm(self.mlp.down_proj(self.pre_feedforward_layernorm(x))))*self.layer_scalar
    cls=type(GEMMA,(torch.nn.Module,),dict(__init__=init,forward=forward))
    torch.manual_seed(13);return cls()
