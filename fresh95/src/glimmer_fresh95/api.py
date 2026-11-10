"""One implementation, two explicit interface identities, unchanged original math."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib,math,torch
from particle_sliders.pg095 import ConditionalPG095
from . import PROFILE,ownership,policy
from .cache import load_cache,to_device
from .util import canonical,equal,new_directory,read_packet,save_packet,write_json

FULL_SCHEMA='Glimmer-shared-explicit-cached-fresh95-FULL-v1'
RAW_SCHEMA='Glimmer-shared-explicit-cached-fresh95-RAW-v1'
META_CONDITIONING=dict(protocol='original-Meta-chat-role-paired-token-v1',original_Meta_chat_roles=True,
    final_neutral_roles=['user'],final_positive_roles=['system','user'],final_history=[],reasoning_strength='low',
    add_generation_prompt=True,token_limit=128,truncation=False)
GLIMMER_CONDITIONING=dict(protocol='original-Glimmer-paired-token-shared-history-v1',student='neutral',teacher='positive',
    continuation='identical-shared-history',truncation=False)

def configuration(value):
    if not isinstance(value,dict):raise ValueError('explicit profile configuration required')
    common={'interface','lane','conditioning','model_identity','width','model_seed','seed','steps','batch','ema_decay',
        'pg_version','pg_source_profile','penalty_input_domain','base_precision','lr','d_lr_mult','prior_lr_mult'}
    lane=value.get('lane');native=lane=='native-nf4'
    if lane not in('cpu-fixture','native-nf4')or set(value)!=(common|({'snapshot','expected_gpu_uuid','execution_profile'}if native else set())):
        raise ValueError('explicit original CPU or native NF4 lane; all configuration/rates required')
    if value.get('interface')not in('glimmer','meta-glimmer'):
        raise ValueError('distinct explicit Glimmer or Meta interface required')
    conditioning=META_CONDITIONING if value['interface']=='meta-glimmer'else GLIMMER_CONDITIONING
    if value.get('conditioning')!=conditioning:raise ValueError('original interface conditioning protocol differs')
    if(value['ema_decay']is not None or value['pg_version']!='0.9.5'or value['pg_source_profile']!='release-0.9.5'
            or value['width']!=(6656 if native else 12)or value['base_precision']!=('bfloat16'if native else'float32')
            or value['penalty_input_domain']!=('continuous'if native else'auto')):
        raise ValueError('exact original lane geometry/precision/PG profile and RAW/noEMA required')
    for key,low,high in(('steps',1,4096),('batch',2,2),('seed',0,2**31-1),('model_seed',0,2**31-1)):
        if type(value[key])is not int or not low<=value[key]<=high:raise ValueError('invalid '+key)
    if native:from glimmer_native.learning_rates import recipe_overrides as rates
    else:from glimmer_modern.learning_rates import recipe_overrides as rates
    if any(value[k]is None for k in('lr','d_lr_mult','prior_lr_mult')):raise ValueError('explicit caller rates; no preset/default migration')
    rates(**{k:value[k]for k in('lr','d_lr_mult','prior_lr_mult')})
    if native:
        from glimmer_native import EXECUTION
        from glimmer_native.contract import canonical_gpu_uuid
        if value['execution_profile']!=EXECUTION or canonical_gpu_uuid(value['expected_gpu_uuid'])!=value['expected_gpu_uuid']:
            raise ValueError('original native physical execution identity required')
        if value['model_seed']!=41:raise ValueError('native original loader seed41 is fixed')
        if not isinstance(value['snapshot'],str)or not Path(value['snapshot']).is_absolute():raise ValueError('original local snapshot path required')
    if not isinstance(value['model_identity'],dict)or not value['model_identity']:raise ValueError('exact original model identity required')
    canonical(value);return deepcopy(value)

def intent(mode,resume,cohort,ema_decay):
    if mode not in('fresh','resume')or not isinstance(cohort,str)or not cohort.strip():raise ValueError('explicit fresh/resume and cohort required')
    if(mode=='fresh'and resume is not None)or(mode=='resume'and resume is None):raise ValueError('fresh refuses state; resume requires exact new FULL')
    if ema_decay is not None:raise ValueError('model-weight EMA refused; RAW only')

def model_gate(model,config):
    if config['lane']=='cpu-fixture':
        from glimmer_modern.native import targets,model_identity
        from glimmer_modern.lifecycle import _model
        targets(model)
        _model({'model_seed':config['model_seed']},model)  # Original reconstructible base-seed guard, unchanged.
        if model_identity(model)!=config['model_identity']:raise ValueError('whole original CPU fixture base differs')
    else:
        from glimmer_native.assets import inspect_snapshot
        from glimmer_native.contract import require_admission,runtime_sources,audit_loaded_model
        from glimmer_native.provenance import verify_sources
        require_admission(config['expected_gpu_uuid']);runtime=runtime_sources()
        assets=inspect_snapshot(config['snapshot'],full_hashes=True)
        identity=audit_loaded_model(model,assets,runtime);_,source=verify_sources();identity['product_source']=source
        if identity!=config['model_identity']:raise ValueError('whole original NF4 asset/runtime/model/numerical identity differs')

class ReceiptBackend:
    def __init__(self,native,receipt,sources,cache_identity,config):
        self.backend=native;self.network=native.network
        self.receipt=deepcopy(receipt);self.sources=deepcopy(sources);self.cache=deepcopy(cache_identity);self.config=deepcopy(config)
    def __getattr__(self,name):return getattr(self.backend,name)
    def metadata(self):
        return dict(self.backend.metadata(),fresh_initialization=dict(profile=PROFILE,receipt=deepcopy(self.receipt),sources=deepcopy(self.sources)),
            interface=self.config['interface'],lane=self.config['lane'],conditioning=deepcopy(self.config['conditioning']),
            completed_native_cache=deepcopy(self.cache),teacher_collector_called=False)

class NativeOwner(ownership.Owner):
    def _components(self,request,component_factory):
        if self.native_lane:
            from glimmer_native.contract import require_admission,runtime_sources
            require_admission(self.config['expected_gpu_uuid']);runtime_sources()
        self.model=component_factory(deepcopy(request));model_gate(self.model,self.config)
        self.attachment=ownership.Attachment(self.model)
        self.native=self.attachment.construct(self.config['lane'],self.config['model_identity'],self.config['seed'])
        self.network=self.native.network
        if self.network.particles is not self.native.slider.particles:raise RuntimeError('original same bank identity lost')
    @contextmanager
    def observation(self,seed):
        from particle_sliders.lifecycle import _network_observation
        self.require_open()
        with _network_observation(self.network,seed,devices=[0]if self.native_lane else[]),self.native.observation():yield
    def response(self,*,ids,strength=1.,seed=7):
        if not math.isfinite(strength)or not 0<=strength<=1:raise ValueError('original unipolar strength within[0,1] required')
        with self.observation(seed),torch.no_grad():return self.native.response(ids=ids,strength=strength,seed=seed)

class Training(NativeOwner):
    def __init__(self,config,cache,directory,*,mode,cohort_id,component_factory,owner_admission,resume=None,ema_decay=None):
        super().__init__();intent(mode,resume,cohort_id,ema_decay)
        if not callable(component_factory)or not callable(owner_admission):raise ValueError('actual owner factory/admission required')
        self.directory=Path(directory).absolute()
        if self.directory.exists():raise FileExistsError('new immutable owner directory required')
        self.config=configuration(config);self.native_lane=self.config['lane']=='native-nf4';self.cohort_id=cohort_id
        self.runtime_identity=ownership.guard(self.config['lane']);self.sources=ownership.sources(self.config['lane'])
        self.options=dict(config=self.config,original_constructor='native-sha256-component-streams'if self.native_lane else'cpu-sha256-component-streams-v1')
        self.construction=policy.Construction(mode,cohort_id,hashlib.sha256(canonical(self.options).encode()).hexdigest())
        packet=read_packet(resume)if mode=='resume'else None
        if packet is not None:
            keys={'schema','profile','interface','lane','cohort_id','options','sources','initialization','cache_identity','game','ema_decay'}
            if(not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=FULL_SCHEMA or packet['profile']!=PROFILE
                    or packet['interface']!=self.config['interface']or packet['lane']!=self.config['lane']
                    or packet['cohort_id']!=cohort_id or packet['options']!=self.options or packet['sources']!=self.sources or packet['ema_decay']is not None):
                raise ValueError('exact whole new-profile role/lane FULL required; no old cohort reset/migration')
            policy.validate_receipt(packet['initialization'],self.construction,self.options,self.sources)
        prepared,conditions,self.cache_identity=load_cache(cache,config=self.config)
        if packet is not None and packet['cache_identity']!=self.cache_identity:raise ValueError('original completed cache differs')
        request=dict(operation=mode,profile=PROFILE,cohort_id=cohort_id,options=deepcopy(self.options),
            completed_cache=deepcopy(self.cache_identity),new_teacher_or_tokenizer_calls=0,
            fresh_PG_initialization_calls=int(mode=='fresh'),ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('owner did not authenticate/admit exact role/lane operation')
        try:
            self._components(request,component_factory)
            if self.native_lane:
                from glimmer_native.native import ReadoutCritic
                self.critic=ReadoutCritic(conditions.shape[1],seed=self.config['seed'])
            else:
                from glimmer_modern.native import ReadoutCritic
                self.critic=ReadoutCritic(12,conditions.shape[1],seed=self.config['seed'])
            self.initialization=(policy.initialize(self.native.slider,self.critic,self.construction,self.options,self.sources)
                if mode=='fresh'else deepcopy(packet['initialization']))
            self.backend=ReceiptBackend(self.native,self.initialization,self.sources,self.cache_identity,self.config)
            device='cuda:0'if self.native_lane else'cpu'
            self.game=ConditionalPG095(self.backend,to_device(prepared,device),self.critic,conditions=conditions.to(device),
                particles=self.native.slider.particles,response_domain='continuous_vector',response_scale=1.0,
                total_steps=self.config['steps'],batch_size=self.config['batch'],seed=self.config['seed'],
                pg_source_profile=self.config['pg_source_profile'],penalty_input_domain=self.config['penalty_input_domain'],
                recipe_overrides={k:self.config[k]for k in('lr','d_lr_mult','prior_lr_mult')})
            if packet is not None:
                self.game.load_state_dict(packet['game'])
                if not equal(self.game.state_dict(),packet['game']):raise RuntimeError('whole game/optimizers/cursors/RNG restore differs')
            self.inherited_steps=self.game.completed_steps
            new_directory(self.directory);write_json(self.directory/'initialization.json',self.initialization)
            self.entry_FULL=self.save(self.directory/f'FULL-{self.inherited_steps:06d}.pt')
            self.entry_RAW=self.export(self.directory/f'RAW-{self.inherited_steps:06d}.pt')
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('Glimmer entry and cleanup failed',[primary,cleanup])
            raise
    def update(self):
        self.require_open()
        if self.native_lane:self.native.validate_numerics()
        return self.game.step()
    def save(self,path):
        self.require_open()
        return save_packet(path,dict(schema=FULL_SCHEMA,profile=PROFILE,interface=self.config['interface'],lane=self.config['lane'],
            cohort_id=self.cohort_id,options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),
            cache_identity=deepcopy(self.cache_identity),game=self.game.state_dict(),ema_decay=None))
    def export(self,path):
        self.require_open();state=self.game.export_state_dict()
        return save_packet(path,dict(schema=RAW_SCHEMA,profile=PROFILE,interface=self.config['interface'],lane=self.config['lane'],
            cohort_id=self.cohort_id,options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),
            cache_identity=deepcopy(self.cache_identity),network=state['network'],completed_steps=self.game.completed_steps,selection='raw',ema_decay=None))

class Inference(NativeOwner):
    def __init__(self,raw,*,component_factory,owner_admission,interface,lane):
        super().__init__();packet=read_packet(raw)
        keys={'schema','profile','interface','lane','cohort_id','options','sources','initialization','cache_identity','network','completed_steps','selection','ema_decay'}
        if(not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=RAW_SCHEMA or packet['profile']!=PROFILE
                or packet['interface']!=interface or packet['lane']!=lane or packet['selection']!='raw'or packet['ema_decay']is not None
                or type(packet['completed_steps'])is not int or packet['completed_steps']<0
                or not callable(component_factory)or not callable(owner_admission)):
            raise ValueError('exact typed new RAW and explicit matching receiving role/lane required')
        self.config=configuration(packet['options']['config']);self.native_lane=self.config['lane']=='native-nf4'
        self.runtime_identity=ownership.guard(self.config['lane']);self.sources=ownership.sources(self.config['lane'])
        self.options=dict(config=self.config,original_constructor='native-sha256-component-streams'if self.native_lane else'cpu-sha256-component-streams-v1')
        if(self.config['interface']!=interface or self.config['lane']!=lane or packet['options']!=self.options
                or packet['sources']!=self.sources or packet['cache_identity']['teacher_identity']!={k:self.config[k]for k in('interface','lane','conditioning','model_identity','width')}
                or packet['completed_steps']>self.config['steps']):raise ValueError('exact role/lane/native/configuration/Source/update identity required')
        construction=policy.Construction('inference',packet['cohort_id'],hashlib.sha256(canonical(self.options).encode()).hexdigest())
        policy.validate_receipt(packet['initialization'],construction,self.options,self.sources)
        self.initialization=deepcopy(packet['initialization'])
        request=dict(operation='inference',profile=PROFILE,cohort_id=packet['cohort_id'],options=deepcopy(self.options),
            completed_cache_metadata_only=deepcopy(packet['cache_identity']),new_teacher_or_tokenizer_calls=0,
            fresh_PG_initialization_calls=0,ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('owner did not admit exact role/lane RAW')
        try:
            self._components(request,component_factory)
            self.network.load_state_dict(packet['network'],strict=True)
            if not equal(self.network.state_dict(),packet['network']):raise RuntimeError('strict native RAW restore differs')
            self.completed_steps=packet['completed_steps']
            # No completed-cache load, teacher/tokenizer, critic/game/optimizer or PG initializer.
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('Glimmer RAW and cleanup failed',[primary,cleanup])
            raise
