"""Second-hop content-head (SCH) amplification at a fixed answer position, with an exact
recomputation ("replay") of the attention residual junction for energy matching.

Energy is measured on the residual state entering the MLP, before its input normalization.
Downstream MLP changes and end-of-block scaling are not part of this estimand.
"""
from contextlib import contextmanager
import torch
from beacon.hooks import _decoder_layers, attention_head_dim
from beacon.pipeline.patching import _encoded_col, _heads_by_layer


def junction(layer):
    if hasattr(layer,'pre_feedforward_layernorm') and hasattr(layer,'post_attention_layernorm'):
        return layer.pre_feedforward_layernorm, layer.post_attention_layernorm
    if hasattr(layer,'post_attention_layernorm') and hasattr(layer,'input_layernorm'):
        return layer.post_attention_layernorm, None
    raise ValueError('Unsupported attention residual junction')


def coordinates(layer,heads,n_heads):
    width=layer.self_attn.o_proj.weight.shape[1];dh=attention_head_dim(layer,n_heads)
    if width != n_heads*dh:raise ValueError('Head dimension does not cover o_proj input')
    if len(set(heads))!=len(heads) or any(h<0 or h>=n_heads for h in heads):raise ValueError('Invalid head IDs')
    return [i for h in heads for i in range(h*dh,(h+1)*dh)]


@torch.inference_mode()
def capture(loaded,prompt,chat,active):
    layers=_decoder_layers(loaded.model);enc,col=_encoded_col(loaded,prompt,chat)
    rows={l:{} for l in active};handles=[]
    for l in active:
        layer=layers[l];entry,_=junction(layer)
        def residual(m,args,l=l):rows[l]['residual']=args[0][:,col:col+1].detach().cpu().clone()
        # Preserve projection GEMM shape: BF16 single-token replay may choose a
        # different kernel, with errors amplified by attention post-normalization.
        def activation(m,args,l=l):
            rows[l]['activation']=args[0].detach().cpu().clone()
            rows[l]['position']=col
        def observed(m,args,l=l):rows[l]['observed']=args[0][:,col:col+1].detach().cpu().clone()
        handles += [layer.input_layernorm.register_forward_pre_hook(residual),
                    layer.self_attn.o_proj.register_forward_pre_hook(activation),entry.register_forward_pre_hook(observed)]
    try:loaded.model(input_ids=enc['input_ids'],use_cache=False)
    finally:
        for h in handles:h.remove()
    if any(set(row)!={'residual','activation','observed','position'} for row in rows.values()):raise ValueError('Attention capture incomplete')
    return rows


class AttentionReplay:
    @torch.inference_mode()
    def __init__(self,layer,rows):
        self.layer=layer;self.rows=rows;self.device=layer.self_attn.o_proj.weight.device
        _,self.norm=junction(layer)
        self.data=[{k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in r.items()} for r in rows]
        self.errors=[];self.single_token_errors=[]
        for row_index,r in enumerate(self.data):
            base=self.output(r['activation'],r['residual'],r.get('position',0));r['base']=base
            err=float((base.float()-r['observed'].float()).square().mean().sqrt()/r['observed'].float().square().mean().sqrt().clamp_min(1e-12))
            if not torch.isfinite(base).all() or err>.005:raise ValueError(f'Attention replay mismatch {err}; row={row_index}, shape={tuple(r["activation"].shape)}, dtype={r["activation"].dtype}')
            self.errors.append(err)
            col=r.get('position',0)
            sliced=self.output(r['activation'][:,col:col+1],r['residual'])
            self.single_token_errors.append(float((sliced.float()-r['observed'].float()).square().mean().sqrt()/r['observed'].float().square().mean().sqrt().clamp_min(1e-12)))
    def output(self,a,residual,position=0):
        projection=self.layer.self_attn.o_proj
        y=torch.nn.functional.linear(a,projection.weight,projection.bias)
        if self.norm is not None:y=self.norm(y)
        return residual+y[:,position:position+1]
    @torch.inference_mode()
    def measure(self,indices,increment):
        values=[]
        for r in self.data:
            a=r['activation'].clone();col=r.get('position',0);a[:,col,indices]*=1+increment
            d=(self.output(a,r['residual'],col).float()-r['base'].float()).square().sum(-1)
            values.append(float(d.mean()))
        return sum(values)/len(values)


@contextmanager
def head_hooks(layers,groups,increments,col,n_heads):
    handles=[]
    try:
        for l,heads in groups.items():
            ix=coordinates(layers[l],heads,n_heads);dose=increments[l]
            def pre(m,args,ix=ix,dose=dose):
                a=args[0].clone();a[:,col,ix]*=1+dose
                return (a,)+tuple(args[1:])
            handles.append(layers[l].self_attn.o_proj.register_forward_pre_hook(pre))
        yield
    finally:
        for h in handles:h.remove()


@torch.inference_mode()
def decode(loaded,prompt,chat,groups,increments,max_new=32):
    enc,col=_encoded_col(loaded,prompt,chat);ids=enc['input_ids'];initial=ids[0].tolist();tokens=[]
    layers=_decoder_layers(loaded.model)
    # Same no-cache greedy argmax loop and tokenizer-EOS stopping rule as SCH amplification.
    with head_hooks(layers,groups,increments,col,loaded.spec.n_heads):
        for _ in range(max_new):
            nxt=int(loaded.model(input_ids=ids,use_cache=False).logits[0,-1].argmax())
            tokens.append(nxt)
            if nxt==loaded.tokenizer.eos_token_id:break
            ids=torch.cat([ids,torch.tensor([[nxt]],device=ids.device)],dim=1)
    return dict(text=loaded.tokenizer.decode(tokens,skip_special_tokens=True),token_ids=tokens,
                prompt_token_ids=initial,position=col,n_generated_tokens=len(tokens),
                stop_reason='tokenizer_eos' if tokens[-1]==loaded.tokenizer.eos_token_id else 'max_new_tokens')


@torch.inference_mode()
def synthetic_test():
    from types import SimpleNamespace
    class Norm(torch.nn.Module):
        def forward(self,x):return x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()
    class Block(torch.nn.Module):
        def __init__(self,postnorm):
            super().__init__();self.input_layernorm=torch.nn.Identity()
            self.self_attn=torch.nn.Module();self.self_attn.o_proj=torch.nn.Linear(8,8,bias=True)
            self.self_attn.head_dim=2
            if postnorm:self.pre_feedforward_layernorm=torch.nn.Identity();self.post_attention_layernorm=Norm()
            else:self.post_attention_layernorm=torch.nn.Identity()
        def forward(self,x):
            a=self.input_layernorm(x);y=self.self_attn.o_proj(a)
            if hasattr(self,'pre_feedforward_layernorm'):return self.pre_feedforward_layernorm(x+self.post_attention_layernorm(y))
            return self.post_attention_layernorm(x+y)
    torch.manual_seed(18)
    for postnorm in (False,True):
        layer=Block(postnorm);x=torch.randn(1,5,8);base=layer(x);col=2
        replay=AttentionReplay(layer,[dict(activation=x,residual=x[:,col:col+1],observed=base[:,col:col+1],position=col)])
        ix=coordinates(layer,[1,3],4)
        with head_hooks([layer],{0:[1,3]},{0:2.},col,4):actual=layer(x)
        expected=float((actual[:,col].float()-base[:,col].float()).square().sum())
        assert abs(replay.measure(ix,2.)-expected)<1e-5
        assert torch.equal(actual[:,[0,1,3,4]],base[:,[0,1,3,4]])
        assert replay.measure(ix,0)==0 and torch.equal(layer(x),base)
        try:
            with head_hooks([layer],{0:[1]},{0:1},col,4):raise RuntimeError('test')
        except RuntimeError:pass
        assert not layer.self_attn.o_proj._forward_pre_hooks
    print('[test] SCH fixed-position, residual/postnorm replay, zero dose and cleanup PASS',flush=True)
