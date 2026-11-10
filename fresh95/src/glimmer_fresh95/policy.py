"""Public PG095 R2 std1 bank and QR down at the original sqrt5 host RMS."""
from dataclasses import dataclass
from importlib import metadata
import hashlib,math
import torch
from torch import nn
from particlegan import init as pg_init
from . import PROFILE
from .util import canonical,source_record

INIT_SHA='3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b'
SCHEMA='Glimmer-explicit-std1-R2bank-sqrt5-downQR-zeroUP-othersKEEP-v1'
MARKER='_glimmer_fresh95_initialization_receipt'

@dataclass(frozen=True)
class Construction:
    mode:str
    cohort:str
    options_sha256:str

class DeclaredOwned(nn.Module):
    def __init__(self,rows):
        super().__init__();self.specs={}
        for i,row in enumerate(rows):
            key='p'+str(i);self.register_parameter(key,row['parameter']);self.specs[key]=row['distribution']

pg_init.register(DeclaredOwned,lambda module:module.specs)

def tensor_sha(value):
    value=value.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()

def seed_for(seed,name):
    return int.from_bytes(hashlib.sha256(f'{PROFILE}:{seed}:{name}'.encode()).digest()[:8],'big')%2**63

def declarations(slider,critic,lane):
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    from hndl.operators.linear import Linear
    if lane=='cpu-fixture':
        from glimmer_modern.native import Slider,ReadoutCritic
    else:
        from glimmer_native.native import Slider,ReadoutCritic
    if type(slider)is not Slider or type(critic)is not ReadoutCritic or not slider.branches:
        raise ValueError('exact original Glimmer HNDL slider and readout required')
    if tuple(slider.particles.shape)!=(128,4):raise ValueError('same original 128x4 bank required')
    rows=[dict(name='slider.particles',parameter=slider.particles,distribution=pg_init.R2Normal(0.,1.),role='std1-R2-particle-bank')]
    for key,branch in slider.branches.items():
        if type(branch)is not HNDLParticleProjection or branch.rank!=8 or branch.scale!=1.:
            raise ValueError('original rank8/alpha8 HNDL particle projection required')
        if branch._particles is not slider.particles:raise ValueError('same original routed bank required')
        down=branch.graph['down'];up=branch.graph['up']
        for leaf in(down,up):
            if (type(leaf)is not Linear or leaf.equalized is not False or leaf.bias is not None
                    or any(leaf.children())or list(leaf.buffers())or nn.utils.parametrize.is_parametrized(leaf)
                    or set(leaf._parameters)!={'weight','bias'}
                    or tuple(leaf.weight.shape)!=(leaf.out_features,leaf.in_features)):
                raise ValueError('original bias-free non-equalized down/up required')
        if down.out_features!=8 or up.in_features!=8 or torch.count_nonzero(up.weight).item():
            raise ValueError('genuine fresh rank8 zero-UP required')
        for name,p in branch.named_parameters(remove_duplicate=False):
            is_down=p is down.weight;is_up=p is up.weight
            bound=math.sqrt(1/down.in_features)  # kaiming_uniform_(a=sqrt5): variance1/(3fan_in).
            rows.append(dict(name='slider.branches.'+key+'.'+name,parameter=p,
                distribution=pg_init.Uniform(-bound,bound)if is_down else pg_init.KEEP,
                role='host-sqrt5-down-QR'if is_down else'protected-zero-UP-KEEP'if is_up else'original-router-bridge-constructor-KEEP'))
    rows.extend(dict(name='critic.'+name,parameter=p,distribution=pg_init.KEEP,
        role='original-readout-constructor-KEEP')for name,p in critic.named_parameters(remove_duplicate=False))
    seen=set();storages=set()
    for row in rows:
        p=row['parameter'];storage=(str(p.device),p.untyped_storage().data_ptr())
        if (type(p)is not nn.Parameter or not p.requires_grad or p.grad is not None
                or p.dtype!=torch.float32 or not p.is_contiguous()or not torch.isfinite(p).all()
                or id(p)in seen or p.numel()and storage in storages or hasattr(p,MARKER)):
            raise ValueError('unique materialized fresh FP32 owned Parameter required')
        seen.add(id(p));storages.add(storage)
    s={id(p)for p in slider.parameters()};c={id(p)for p in critic.parameters()}
    if seen!=s|c or s&c:raise ValueError('declare only and all owned slider/critic leaves once')
    return rows

def initialize(slider,critic,construction,options,sources):
    if (not isinstance(construction,Construction)or construction.mode!='fresh'or not construction.cohort
            or construction.options_sha256!=hashlib.sha256(canonical(options).encode()).hexdigest()):
        raise ValueError('explicit fresh construction before optimizers required')
    if metadata.version('particlegan')!='0.9.5'or source_record(pg_init.__file__)['sha256']!=INIT_SHA:
        raise ValueError('exact public PG095 initializer required')
    config=options['config'];rows=declarations(slider,critic,config['lane'])
    before=[tensor_sha(row['parameter'])for row in rows];ids=[id(row['parameter'])for row in rows]
    holder=DeclaredOwned(rows);seeds={'p'+str(i):seed_for(config['seed'],row['name'])for i,row in enumerate(rows)}
    pg_init.deterministic_orthogonal_(holder,strict=True,parameter_seeds=seeds)
    if ids!=[id(row['parameter'])for row in rows]:raise RuntimeError('owned Parameter replaced')
    parameters=[]
    for i,row in enumerate(rows):
        after=tensor_sha(row['parameter'])
        if row['distribution']is pg_init.KEEP and before[i]!=after:raise RuntimeError('deliberate original constructor changed')
        parameters.append(dict(name=row['name'],role=row['role'],seed=seeds['p'+str(i)],shape=list(row['parameter'].shape),
            dtype=str(row['parameter'].dtype),distribution=repr(row['distribution']),before_sha256=before[i],after_sha256=after))
    receipt=dict(schema=SCHEMA,profile=PROFILE,interface=config['interface'],lane=config['lane'],cohort=construction.cohort,
        options_sha256=construction.options_sha256,source_identity=sources,seed=config['seed'],completed_steps=0,
        restored_from=None,initializer=source_record(pg_init.__file__),parameters=parameters,parameter_identity_preserved=True,
        particle_bank='original-same-Parameter-128x4-R2Normal(0,1)',host_down_kaiming_a='sqrt5',
        host_down_target_mean_square='1/(3fan_in)',other_adapter_and_critic='constructor-KEEP',selection='raw',ema_decay=None)
    marker=hashlib.sha256(canonical(receipt).encode()).hexdigest()
    for row in rows:setattr(row['parameter'],MARKER,marker)
    return receipt

def validate_receipt(receipt,construction,options,sources):
    config=options['config']
    if (not isinstance(receipt,dict)or receipt.get('schema')!=SCHEMA or receipt.get('profile')!=PROFILE
            or receipt.get('interface')!=config['interface']or receipt.get('lane')!=config['lane']
            or receipt.get('cohort')!=construction.cohort or receipt.get('options_sha256')!=construction.options_sha256
            or receipt.get('source_identity')!=sources or receipt.get('seed')!=config['seed']
            or receipt.get('completed_steps')!=0 or receipt.get('restored_from')is not None
            or receipt.get('ema_decay')is not None or receipt.get('selection')!='raw'
            or receipt.get('initializer',{}).get('sha256')!=INIT_SHA):
        raise ValueError('exact lane/interface/cohort new-profile initialization receipt required')
