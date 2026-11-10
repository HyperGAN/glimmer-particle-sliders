"""FOUR NEW UNRUN shared CPU cases, ONE total update; native NF4 unqualified.

Both interface boundaries are exercised using original width12 causal model
and synthetic completed fields. No old test/cache/teacher/tokenizer is used.
"""
from copy import deepcopy
import math
import pytest,torch
from particle_sliders.pg09_audio import AudioRow,PreparedAudioResponse,ARLastHiddenResponseAdapter
from glimmer_modern import lifecycle as original
from glimmer_modern.fixture import make_model
from glimmer_modern.native import model_identity
from glimmer_fresh95 import open_training,open_inference,save_prepared_cache,file_pin,Checkpoint
from glimmer_fresh95 import api,cache,policy,ownership
from glimmer_fresh95.util import equal,read_packet,save_packet

def component_factory(request):
    assert request['operation']in('fresh','resume','inference')
    assert request['options']['config']['lane']=='cpu-fixture'
    assert request['new_teacher_or_tokenizer_calls']==0
    return make_model(request['options']['config']['model_seed'])

@pytest.fixture
def staged(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('no original prepare/read_rows/teacher/tokenizer or pretrained loader')
    monkeypatch.setattr(original,'prepare',forbidden);monkeypatch.setattr(original,'read_rows',forbidden)
    monkeypatch.setattr(ARLastHiddenResponseAdapter,'prepare',forbidden)
    model=make_model(41);base=deepcopy(model.state_dict());identity=model_identity(model)
    prepared=[PreparedAudioResponse(AudioRow((1,2+i),(5,7+i),0,1.),
        torch.linspace(-.2,.3,12).reshape(1,12)+i*.02,
        torch.linspace(-.1,.1,12).reshape(1,12))for i in range(3)]
    data=tmp_path/'NEW-synthetic-completed-fields.pt';torch.save([cache.dump(x)for x in prepared],data)
    whole=file_pin(data);source=file_pin(__file__);options={}
    for interface in('glimmer','meta-glimmer'):
        conditioning=deepcopy(api.META_CONDITIONING if interface=='meta-glimmer'else api.GLIMMER_CONDITIONING)
        config=dict(interface=interface,lane='cpu-fixture',conditioning=conditioning,model_identity=identity,width=12,
            model_seed=41,seed=7,steps=3,batch=2,ema_decay=None,pg_version='0.9.5',pg_source_profile='release-0.9.5',
            penalty_input_domain='auto',base_precision='float32',lr=.001,d_lr_mult=18.,prior_lr_mult=30.)
        origins=[dict(scope='synthetic-software-only',complete_prepared_output=True,synthetic_row=i,
            interface=interface,lane='cpu-fixture',conditioning=conditioning,
            whole_output_files=[dict(path=whole.path,bytes=whole.bytes,sha256=whole.sha256)],
            source_files=[dict(path=source.path,bytes=source.bytes,sha256=source.sha256)],real_teacher_calls=0)for i in range(3)]
        frozen=save_prepared_cache(tmp_path/(interface+'-completed-cache.pt'),config=config,prepared=prepared,
            teacher_origins=origins,cache_id='NEW-'+interface+'-synthetic-completed-fields',scope='synthetic-software-only',
            owner_admission=lambda request:True)
        options[interface]=dict(config=config,cache=frozen,cohort_id='NEW-shared-software-'+interface,
            component_factory=component_factory,owner_admission=lambda request:True)
    return options,tmp_path,base

def opened(staged,interface,name):
    options,root,_=staged
    return open_training(directory=root/name,mode='fresh',**options[interface])

def test_new_both_interfaces_same_Parameters_R2bank_sqrt5down_zeroUP_KEEP_before_optimizers(staged,monkeypatch):
    initialize=policy.initialize;game=api.ConditionalPG095;seen=[]
    def initialize_spy(slider,critic,*args):
        ids=[id(p)for p in slider.parameters()]+[id(p)for p in critic.parameters()]
        before=deepcopy(critic.state_dict());receipt=initialize(slider,critic,*args)
        assert ids==[id(p)for p in slider.parameters()]+[id(p)for p in critic.parameters()]
        assert equal(before,critic.state_dict())
        assert all(torch.count_nonzero(branch.graph['up'].weight)==0 for branch in slider.branches.values())
        assert any(row['distribution']=='R2Normal(mean=0.0, std=1.0)'for row in receipt['parameters'])
        assert all(row['before_sha256']==row['after_sha256']for row in receipt['parameters']if row['role'].endswith('KEEP'))
        seen.append(receipt);return receipt
    def construct(backend,*args,**kwargs):
        assert seen and backend.metadata()['fresh_initialization']['receipt']==seen[-1]
        assert kwargs['particles']is backend.backend.slider.particles is backend.network.particles
        assert kwargs['response_scale']==1.0
        return game(backend,*args,**kwargs)
    monkeypatch.setattr(policy,'initialize',initialize_spy);monkeypatch.setattr(api,'ConditionalPG095',construct)
    for interface in('glimmer','meta-glimmer'):
        with opened(staged,interface,'before-'+interface)as owner:
            assert owner.game.completed_steps==0 and owner.game.prior is not None
            assert owner.game.state_dict()['config']['backend']['fresh_initialization']['receipt']==seen[-1]
            assert owner.initialization['interface']==owner.cache_identity['teacher_identity']['interface']==interface
            assert owner.options['config']['conditioning']==(api.META_CONDITIONING if interface=='meta-glimmer'else api.GLIMMER_CONDITIONING)
            assert (owner.directory/'FULL-000000.pt').is_file()and(owner.directory/'RAW-000000.pt').is_file()
            assert {id(p)for p in owner.network.parameters()}=={id(p)for p in owner.native.slider.parameters()}
    assert seen[0]['interface']!=seen[1]['interface']and seen[0]['cohort']!=seen[1]['cohort']

def test_new_R2_quantile_sqrt5_Gram_determinism_RNG_frozen_base_both_interfaces(staged):
    before=torch.get_rng_state().clone();_,_,base=staged;states=[]
    for interface in('glimmer','meta-glimmer'):
        with opened(staged,interface,'laws-'+interface)as owner:
            state=deepcopy(owner.network.state_dict());states.append(state);model=owner.model;attachment=owner.attachment
            expected=torch.special.ndtri(policy.pg_init._qr.r2_points(128,4)).to(dtype=torch.float32)
            assert torch.equal(owner.native.slider.particles.detach().cpu(),expected)
            for branch in owner.native.slider.branches.values():
                down=branch.graph['down'].weight.detach();rows,cols=down.shape;variance=1/(3*cols)
                gram=down@down.T if rows<=cols else down.T@down
                assert torch.allclose(gram,torch.eye(min(rows,cols))*variance*max(rows,cols),rtol=1e-5,atol=1e-6)
                assert math.isclose(float(down.square().mean()),variance,rel_tol=1e-5,abs_tol=1e-7)
                assert torch.count_nonzero(branch.graph['up'].weight)==0
            assert equal(base,model.state_dict())
        assert all(p.requires_grad==flag for p,flag in attachment.flags)
        assert all(m.training==mode for m,mode in attachment.modes)
        assert not hasattr(model,ownership.MARKER)
        assert not any('forward'in vars(m)or'_glimmer_modern'in vars(m)for m in model.modules())
        assert torch.equal(before,torch.get_rng_state())
    assert equal(states[0],states[1])  # Same common base law, separate role receipts.
    with opened(staged,'glimmer','deterministic-again')as again:assert equal(states[0],again.network.state_dict())
    assert torch.equal(before,torch.get_rng_state())

def test_new_ONE_update_whole_FULL_RAW_skip_and_cross_interface_lane_refusal(staged,monkeypatch):
    options,root,base=staged
    initializer=policy.initialize;game=api.ConditionalPG095;cache_loader=api.load_cache
    with opened(staged,'glimmer','one-update')as trained:
        trained.update()  # The ONE total new software update, common base only.
        full=trained.save(root/'FULL1.pt');raw=trained.export(root/'RAW1.pt')
        state=deepcopy(trained.game.state_dict());network=deepcopy(trained.network.state_dict());model=trained.model
    assert equal(base,model.state_dict())
    def forbidden(*args,**kwargs):raise AssertionError('old cache/teacher/initializer or wrong-boundary factory called')
    for changed in('interface','lane'):
        packet=read_packet(full);packet[changed]='meta-glimmer'if changed=='interface'else'native-nf4'
        bad=save_packet(root/('wrong-'+changed+'-FULL.pt'),packet)
        with pytest.raises(ValueError):open_training(directory=root/('wrong-'+changed),mode='resume',resume=bad,
            **dict(options['glimmer'],component_factory=forbidden))
    with pytest.raises(ValueError):open_training(directory=root/'cross-meta',mode='resume',resume=full,
        **dict(options['meta-glimmer'],component_factory=forbidden))
    for interface,lane in(('meta-glimmer','cpu-fixture'),('glimmer','native-nf4')):
        with pytest.raises(ValueError):open_inference(raw,interface=interface,lane=lane,
            component_factory=forbidden,owner_admission=lambda request:True)
    monkeypatch.setattr(policy,'initialize',forbidden)
    with open_training(directory=root/'resume',mode='resume',resume=full,**options['glimmer'])as resumed:
        assert resumed.inherited_steps==1 and equal(state,resumed.game.state_dict())
    monkeypatch.setattr(api,'load_cache',forbidden);monkeypatch.setattr(api,'ConditionalPG095',forbidden)
    with open_inference(raw,interface='glimmer',lane='cpu-fixture',component_factory=component_factory,owner_admission=lambda request:True)as inferred:
        assert not hasattr(inferred,'game')and not hasattr(inferred,'critic')
        assert equal(network,inferred.network.state_dict())
        before=torch.get_rng_state().clone();value=inferred.response(ids=(1,2),strength=.5,seed=17)
        assert value.shape==(1,12)and torch.isfinite(value).all()and torch.equal(before,torch.get_rng_state())
        assert equal(base,inferred.model.state_dict())
    # A Meta RAW from the same common implementation is independently bound;
    # no second update or old evidence is required to check its reload boundary.
    monkeypatch.setattr(policy,'initialize',initializer);monkeypatch.setattr(api,'load_cache',cache_loader);monkeypatch.setattr(api,'ConditionalPG095',game)
    with opened(staged,'meta-glimmer','meta-zero-raw')as meta:meta_raw=meta.export(root/'Meta-RAW0.pt')
    monkeypatch.setattr(policy,'initialize',forbidden);monkeypatch.setattr(api,'load_cache',forbidden);monkeypatch.setattr(api,'ConditionalPG095',forbidden)
    with open_inference(meta_raw,interface='meta-glimmer',lane='cpu-fixture',component_factory=component_factory,owner_admission=lambda request:True)as meta:
        assert meta.completed_steps==0 and meta.config['conditioning']==api.META_CONDITIONING

def test_new_both_role_early_refusals_foreign_marker_callbacks_attributes_flags_modes_retry(staged):
    options,root,_=staged;calls=[]
    def forbidden(request):calls.append(request);raise AssertionError('invalid intent/cache reached model factory')
    for interface in('glimmer','meta-glimmer'):
        bad=dict(options[interface],component_factory=forbidden)
        for mode,resume,ema in(('fresh',Checkpoint('unused',1,'0'*64),None),('resume',None,None),('fresh',None,.9)):
            with pytest.raises(ValueError):open_training(directory=root/('bad-'+interface),mode=mode,resume=resume,ema_decay=ema,**bad)
        wrong_cache=options['meta-glimmer'if interface=='glimmer'else'glimmer']['cache']
        with pytest.raises(ValueError):open_training(directory=root/('wrong-cache-'+interface),mode='fresh',**dict(bad,cache=wrong_cache))
        cfg=deepcopy(bad['config']);cfg['ema_decay']=.9
        with pytest.raises(ValueError):open_training(directory=root/('ema-'+interface),mode='fresh',**dict(bad,config=cfg))
    assert calls==[]
    owner=opened(staged,'meta-glimmer','owned-cleanup');attachment=owner.attachment;model=owner.model
    # Synthetic callback/checkpoint ledgers exercise cleanup only; this does
    # not execute checkpointing or claim a native NF4 forward observation.
    module=model.model.language_model.layers[0];callback=lambda *args:None
    handle=module.register_forward_pre_hook(callback);container=[handle]
    owner.native._test_owned_handles=container
    hooks=handle.hooks_dict_ref();attachment.containers.append((owner.native,'_test_owned_handles',container,(handle,)))
    attachment.pending_hooks.append((handle,hooks,callback,()))
    checkpoint=object();module._test_owned_checkpoint=checkpoint
    attachment.pending_attrs.append((module,'_test_owned_checkpoint',False,None,True,checkpoint))
    extra=object();container.append(extra)
    with pytest.raises(RuntimeError):owner.require_open()
    container.pop()
    other_callback=lambda *args:None;hooks[handle.id]=other_callback
    with pytest.raises(RuntimeError):owner.require_open()
    hooks[handle.id]=callback
    projection,key,_,_,_,owned=next(value for value in attachment.pending_attrs if value[1]=='forward')
    projection.forward=lambda *args:None
    with pytest.raises(RuntimeError):owner.require_open()
    setattr(projection,key,owned)
    parameter=attachment.flags[0][0];parameter.requires_grad_(True)
    with pytest.raises(RuntimeError):owner.require_open()
    parameter.requires_grad_(attachment.owned_flags[id(parameter)])
    module.training=True
    with pytest.raises(RuntimeError):owner.require_open()
    module.training=attachment.owned_modes[id(module)]
    owner.require_open()
    flags=[(p,p.requires_grad)for p,_ in attachment.flags];modes=[(m,m.training)for m,_ in attachment.modes]
    attrs=[(m,k,k in vars(m),vars(m).get(k))for m,k,*_ in attachment.pending_attrs]
    foreign=object();setattr(model,ownership.MARKER,foreign)
    with pytest.raises(RuntimeError):owner.close()
    assert vars(model)[ownership.MARKER]is foreign and container==[handle]and hooks[handle.id]is callback
    assert all(p.requires_grad is flag for p,flag in flags)and all(m.training is mode for m,mode in modes)
    assert all((k in vars(m))==had and vars(m).get(k)is value for m,k,had,value in attrs)
    setattr(model,ownership.MARKER,attachment)
    foreign_callback=lambda *args:None;hooks[handle.id]=foreign_callback
    foreign_checkpoint=object();module._test_owned_checkpoint=foreign_checkpoint
    with pytest.raises(BaseExceptionGroup):owner.close()
    assert hooks[handle.id]is foreign_callback and module._test_owned_checkpoint is foreign_checkpoint
    assert vars(model)[ownership.MARKER]is attachment and container==[handle]
    hooks[handle.id]=callback;module._test_owned_checkpoint=checkpoint
    parameter.requires_grad_(True)
    # Original fixture eval=False means True is a foreign mode even though
    # it can equal a caller's prior flag; compare captured owned, not old value.
    module.training=True
    with pytest.raises(BaseExceptionGroup):owner.close()
    assert parameter.requires_grad is True and module.training is True
    assert any(p is parameter for p,_ in attachment.pending_flags)and any(m is module for m,_ in attachment.pending_modes)
    assert vars(model)[ownership.MARKER]is attachment and not attachment.pending_hooks and not attachment.pending_attrs
    parameter.requires_grad_(attachment.owned_flags[id(parameter)]);module.training=attachment.owned_modes[id(module)]
    owner.close();owner.close()
    assert owner.closed and not hasattr(model,ownership.MARKER)and not hooks and container==[]
    assert all(p.requires_grad==flag for p,flag in attachment.flags)and all(m.training==mode for m,mode in attachment.modes)
    with pytest.raises(RuntimeError):owner.update()  # Refused after close; no second update.
