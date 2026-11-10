"""Exact original guards and constructor-owned, foreign-safe borrowed cleanup."""
from importlib import metadata
from pathlib import Path
import base64,csv,hashlib,io
from .baseline import MODERN,NATIVE
from .util import source_record

MARKER='_explicit_shared_glimmer_fresh95_owner'
ATTRS=('forward','_glimmer_modern','_glimmer_native','_pg09_scale_checkpoint',
       'gradient_checkpointing','_gradient_checkpointing_func')

def distribution(name,version):
    normalize=lambda x:x.lower().replace('_','-')
    rows=[d for d in metadata.distributions()if normalize(d.metadata.get('Name')or'')==normalize(name)]
    if len(rows)!=1 or rows[0].version!=version:raise RuntimeError('one exact owning distribution required:'+name)
    return rows[0]

def record_guard(dist,namespace,root,pins):
    if Path(dist.locate_file(namespace+'/__init__.py')).absolute()!=root/'__init__.py':
        raise RuntimeError('original/addon import shadowed from genuine distribution:'+namespace)
    records={row[0]:row[1:]for row in csv.reader(io.StringIO(dist.read_text('RECORD')or''))}
    for name,pin in pins.items():
        path=root/name
        if source_record(path)!=pin:raise RuntimeError('captured original Source differs:'+name)
        raw=path.read_bytes();encoded=base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode()
        if records.get(namespace+'/'+name)!=['sha256='+encoded,str(len(raw))]:
            raise RuntimeError('genuine owning RECORD differs:'+namespace+'/'+name)

def guard(lane):
    from particlegan import init as pg_init
    from .policy import INIT_SHA
    if lane=='cpu-fixture':
        import glimmer_modern as original
        from glimmer_modern.provenance import verify_sources
        name,version,pins='meta-glimmer-modern-response','0.2.0',MODERN
    elif lane=='native-nf4':
        import glimmer_native as original
        from glimmer_native.provenance import verify_sources
        name,version,pins='glimmer-native-response-candidate','0.2.0',NATIVE
    else:raise ValueError('explicit original lane required')
    record_guard(distribution(name,version),original.__name__,Path(original.__file__).absolute().parent,pins)
    _,source=verify_sources()  # Original recursive core577; native also product14, unmodified.
    if metadata.version('particlegan')!='0.9.5'or source_record(pg_init.__file__)['sha256']!=INIT_SHA:
        raise RuntimeError('exact public PG095 initializer required')
    own=Path(__file__).absolute().parent
    record_guard(distribution('glimmer-explicit-fresh95','0.1.0'),'glimmer_fresh95',own,
        {p.name:source_record(p)for p in sorted(own.glob('*.py'))})
    return dict(lane=lane,original_distribution=name,original_version=version,original_source=source,
        original_core_and_native_recursive_guards=True,addon_genuine_RECORD=True)

def sources(lane):
    import importlib
    from particle_sliders import pg09,pg095,pg09_audio,pg09_backends,lifecycle,fibo_hndl
    prefix='glimmer_modern'if lane=='cpu-fixture'else'glimmer_native'
    modules=[importlib.import_module(prefix+'.'+name)for name in('native','provenance','learning_rates')]
    modules.extend((pg09,pg095,pg09_audio,pg09_backends,lifecycle,fibo_hndl))
    return dict(lane=lane,own={p.name:source_record(p)for p in sorted(Path(__file__).absolute().parent.glob('*.py'))},
        original={m.__name__:source_record(m.__file__)for m in modules})

class Attachment:
    """Only the bounded original Native constructor may create owned deltas.

    No old Native/Slider.close is delegated: those original broad cleanups
    restore flags even after foreign ownership changed. This ledger snapshots
    raw instance ownership before construction and the exact final/partial
    constructor state, then retries only pending private operations.
    """
    def __init__(self,model):
        if hasattr(model,MARKER):raise ValueError('another shared Glimmer owner exists')
        self.model=model;self.native=None;self.closed=self.closing=False
        self.modules=tuple(model.modules());self.flags=[(p,p.requires_grad)for p in model.parameters()]
        self.modes=[(m,m.training)for m in self.modules]
        self.before=[(m,key,key in vars(m),vars(m).get(key))for m in self.modules for key in ATTRS]
        self.pending_attrs=[];self.pending_hooks=[];self.containers=[]
        self.pending_flags=list(self.flags);self.pending_modes=list(self.modes);self.pending_marker=True
        self.owned_flags={id(p):flag for p,flag in self.flags};self.owned_modes={id(m):mode for m,mode in self.modes}
        setattr(model,MARKER,self)

    def construct(self,lane,identity,seed):
        if lane=='cpu-fixture':from glimmer_modern.native import Native
        else:from glimmer_native.native import Native
        self.native=Native.__new__(Native)
        try:
            if lane=='cpu-fixture':Native.__init__(self.native,self.model,seed=seed)
            else:Native.__init__(self.native,self.model,identity,seed=seed)
        finally:self._capture_constructor_state()
        self.ids=[(n,id(p))for n,p in self.native.network.named_parameters()]
        return self.native

    def _capture_constructor_state(self):
        # Never called on a user operation or cleanup retry: foreign settings
        # cannot be laundered into this constructor's owned state.
        self.pending_attrs=[(m,key,had,old,key in vars(m),vars(m).get(key))for m,key,had,old in self.before
            if(key in vars(m))!=had or vars(m).get(key)is not old]
        self.owned_flags={id(p):p.requires_grad for p,_ in self.flags}
        self.owned_modes={id(m):m.training for m,_ in self.modes}
        container=getattr(self.native,'_forward_guards',None)
        if container is not None:
            if type(container)is not list:raise RuntimeError('original native private guard list required')
            items=tuple(container);self.containers.append((self.native,'_forward_guards',container,items))
            for handle in items:
                hooks=handle.hooks_dict_ref()
                if hooks is None or handle.id not in hooks:raise RuntimeError('original private callback missing')
                extras=[]
                for reference in getattr(handle,'extra_dict_ref',()):
                    extra=reference()
                    if extra is not None:extras.append((extra,handle.id in extra,extra.get(handle.id)))
                self.pending_hooks.append((handle,hooks,hooks[handle.id],tuple(extras)))
        backend=getattr(self.native,'backend',None)
        checkpoints=getattr(backend,'_checkpointing',None)
        if checkpoints is not None:
            if type(checkpoints)is not list:raise RuntimeError('original private checkpoint list required')
            self.containers.append((backend,'_checkpointing',checkpoints,tuple(checkpoints)))

    def _containers_unchanged(self):
        for owner,key,container,items in self.containers:
            current=getattr(owner,key,None)
            if current is not container or len(current)!=len(items)or any(a is not b for a,b in zip(current,items)):
                raise RuntimeError('foreign private handle container/items retained')

    def require_active(self):
        if self.closed or self.closing:raise RuntimeError('Glimmer attachment closed/cleanup pending')
        if vars(self.model).get(MARKER)is not self:raise RuntimeError('foreign shared Glimmer marker')
        if [(n,id(p))for n,p in self.native.network.named_parameters()]!=self.ids:raise RuntimeError('original adapter/bank Parameter identity changed')
        if any(p.requires_grad is not self.owned_flags[id(p)]for p,_ in self.flags):raise RuntimeError('frozen host flag changed')
        if any(m.training is not self.owned_modes[id(m)]for m,_ in self.modes):raise RuntimeError('native mode ownership changed')
        self._containers_unchanged()
        for module,key,_,_,has_owned,owned in self.pending_attrs:
            if(key in vars(module))!=has_owned or vars(module).get(key)is not owned:
                raise RuntimeError('foreign native/checkpoint attribute before operation:'+key)
        for handle,hooks,callback,extras in self.pending_hooks:
            if handle.hooks_dict_ref()is not hooks or hooks.get(handle.id)is not callback:
                raise RuntimeError('foreign native guard callback before operation')
            if any((handle.id in extra)!=had or extra.get(handle.id)is not owned for extra,had,owned in extras):
                raise RuntimeError('foreign native guard extra-dictionary entry before operation')

    @staticmethod
    def _run(pending,operation,errors):
        failed=[]
        for value in pending:
            try:operation(value)
            except BaseException as error:failed.append(value);errors.append(error)
        return failed

    def close(self):
        if self.closed:return
        self.closing=True
        if not self.pending_marker or vars(self.model).get(MARKER)is not self:
            raise RuntimeError('foreign Glimmer marker retained before any borrowed-state cleanup')
        errors=[]
        try:self._containers_unchanged()
        except BaseException as error:errors.append(error)
        else:
            def hook(value):
                handle,hooks,callback,extras=value
                if handle.hooks_dict_ref()is not hooks or hooks.get(handle.id)is not callback:
                    raise RuntimeError('foreign native guard callback retained')
                if any((handle.id in extra)!=had or extra.get(handle.id)is not owned for extra,had,owned in extras):
                    raise RuntimeError('foreign native guard extra-dictionary entry retained')
                handle.remove()
            self.pending_hooks=self._run(self.pending_hooks,hook,errors)
            def attribute(value):
                module,key,had,old,has_owned,owned=value
                if(key in vars(module))!=has_owned or vars(module).get(key)is not owned:
                    raise RuntimeError('foreign original native/checkpoint attribute retained:'+key)
                if had:setattr(module,key,old)
                else:vars(module).pop(key,None)
            self.pending_attrs=self._run(self.pending_attrs,attribute,errors)
        if not self.pending_hooks and not self.pending_attrs and not errors:
            # Private list contents remain untouched until every borrowed hook
            # and attribute is restored; a foreign append/replacement is refused.
            self._containers_unchanged()
            for _,_,container,_ in self.containers:container.clear()
            self.containers=[]
            def flag(value):
                parameter,original=value
                if parameter.requires_grad is not self.owned_flags[id(parameter)]:raise RuntimeError('foreign native gradient flag retained')
                parameter.requires_grad_(original)
            def mode(value):
                module,original=value
                if module.training is not self.owned_modes[id(module)]:raise RuntimeError('foreign native mode retained')
                module.training=original
            self.pending_flags=self._run(self.pending_flags,flag,errors)
            self.pending_modes=self._run(self.pending_modes,mode,errors)
            if not self.pending_flags and not self.pending_modes:
                if vars(self.model).get(MARKER)is not self:errors.append(RuntimeError('foreign Glimmer marker retained'))
                else:delattr(self.model,MARKER);self.pending_marker=False
        if errors:raise BaseExceptionGroup('Glimmer cleanup pending',errors)
        if any((self.pending_attrs,self.pending_hooks,self.pending_flags,self.pending_modes,self.containers))or self.pending_marker:
            raise RuntimeError('Glimmer cleanup pending')
        self.closed,self.closing=True,False

class Owner:
    def __init__(self):self.closed=self.closing=False;self.attachment=None
    def require_open(self):
        if self.closed or self.closing:raise RuntimeError('Glimmer owner closed/cleanup pending')
        if self.attachment is not None:self.attachment.require_active()
    def close(self):
        if self.closed:return
        self.closing=True
        if self.attachment is not None:self.attachment.close()
        self.closed,self.closing=True,False
    def __enter__(self):self.require_open();return self
    def __exit__(self,kind,primary,traceback):
        try:self.close()
        except BaseException as cleanup:
            if primary is not None:raise BaseExceptionGroup('Glimmer work and cleanup failed',[primary,cleanup])
            raise
