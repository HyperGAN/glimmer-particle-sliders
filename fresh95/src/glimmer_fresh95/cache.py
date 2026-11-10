"""Completed original AudioRows/last-hidden fields only; no teacher collection."""
from dataclasses import fields
from pathlib import Path
import torch
from particle_sliders.pg09_audio import AudioRow,PreparedAudioResponse,audio_conditions
from . import PROFILE,ownership
from .util import canonical,equal,read_packet,save_packet,source_record,tree

SCHEMA='Glimmer-shared-completed-native-PreparedAudioResponses-fresh95-v1'

def file_record(value):
    if (not isinstance(value,dict)or set(value)!={'path','bytes','sha256'}
            or not isinstance(value['path'],str)or not Path(value['path']).is_absolute()or'..'in Path(value['path']).parts
            or type(value['bytes'])is not int or value['bytes']<1
            or not isinstance(value['sha256'],str)or len(value['sha256'])!=64
            or any(c not in'0123456789abcdef'for c in value['sha256'])):
        raise ValueError('whole original file byte/SHA origin required')

def validate_prepared(prepared,config):
    if not isinstance(prepared,(list,tuple))or len(prepared)<2:raise ValueError('at least2 completed original PreparedAudioResponses required')
    if config['lane']=='cpu-fixture':from glimmer_modern.native import validate_ids
    else:from glimmer_native.contract import validate_ids
    for item in prepared:
        if type(item)is not PreparedAudioResponse or type(item.row)is not AudioRow:
            raise ValueError('actual original PreparedAudioResponse/AudioRow required')
        row=item.row
        if type(row.neutral_ids)is not tuple or type(row.positive_ids)is not tuple or row.scale!=1.:
            raise ValueError('whole original fixed tuple IDs and unipolar unit target required')
        validate_ids(row.neutral_ids);validate_ids(row.positive_ids)
        if row.neutral_ids==row.positive_ids or type(row.history_tokens)is not int or not 0<=row.history_tokens<min(len(row.neutral_ids),len(row.positive_ids)):
            raise ValueError('distinct original prompts and explicit shared continuation required')
        if row.history_tokens and row.neutral_ids[-row.history_tokens:]!=row.positive_ids[-row.history_tokens:]:
            raise ValueError('original teacher/student history bytes differ')
        if config['interface']=='meta-glimmer'and row.history_tokens!=0:
            raise ValueError('original Meta final conditioning uses empty history')
        for value in(item.target,item.neutral):
            if (type(value)is not torch.Tensor or value.shape!=(1,config['width'])or value.device.type!='cpu'
                    or value.dtype!=torch.float32 or value.requires_grad or value.layout!=torch.strided
                    or not torch.isfinite(value).all()):
                raise ValueError('complete frozen finite original CPU FP32 last-hidden fields required')

def dump(item):
    return dict(row={f.name:getattr(item.row,f.name)for f in fields(AudioRow)},target=tree(item.target,'cpu'),neutral=tree(item.neutral,'cpu'))

def restore(value,device):
    if(not isinstance(value,dict)or set(value)!={'row','target','neutral'}or not isinstance(value['row'],dict)
            or set(value['row'])!={f.name for f in fields(AudioRow)}):raise ValueError('complete original serialized AudioRow required')
    return PreparedAudioResponse(AudioRow(**value['row']),tree(value['target'],device),tree(value['neutral'],device))

def teacher_identity(config):
    return {key:config[key]for key in('interface','lane','conditioning','model_identity','width')}

def origins_valid(origins,prepared,config,scope):
    if not isinstance(origins,list)or len(origins)!=len(prepared):raise ValueError('each completed row requires authenticated whole-output origins')
    for origin in origins:
        if (not isinstance(origin,dict)or origin.get('scope')!=scope or origin.get('complete_prepared_output')is not True
                or origin.get('interface')!=config['interface']or origin.get('lane')!=config['lane']
                or origin.get('conditioning')!=config['conditioning']
                or not isinstance(origin.get('whole_output_files'),list)or not origin['whole_output_files']
                or not isinstance(origin.get('source_files'),list)or not origin['source_files']):
            raise ValueError('complete role/lane/conditioning-bound original output and Source provenance required')
        for record in origin['whole_output_files']+origin['source_files']:file_record(record)
    canonical(origins)

def save_prepared_cache(path,*,config,prepared,teacher_origins,cache_id,scope,owner_admission):
    from .api import configuration
    config=configuration(config);ownership.guard(config['lane']);validate_prepared(prepared,config)
    if scope not in('completed-native','synthetic-software-only')or(config['lane']=='native-nf4'and scope!='completed-native'):
        raise ValueError('synthetic CPU and authentic native completed origins remain distinct')
    origins_valid(teacher_origins,prepared,config,scope)
    if not isinstance(cache_id,str)or not cache_id.strip()or not callable(owner_admission):raise ValueError('explicit cache ID and owner required')
    request=dict(operation='serialize-completed-cache',profile=PROFILE,cache_id=cache_id,scope=scope,
        teacher_identity=teacher_identity(config),teacher_origins=teacher_origins,model_or_teacher_calls=0)
    if owner_admission(request)is not True:raise ValueError('owner did not authenticate completed output and original token/template origin')
    # Owner authenticates original saved output and role/template provenance.
    # This binds already supplied tensors and original pure audio_conditions;
    # no tokenizer, model, prepare, teacher or cache generation is called here.
    return save_packet(path,dict(schema=SCHEMA,profile=PROFILE,cache_id=cache_id,scope=scope,
        teacher_identity=teacher_identity(config),teacher_origins=teacher_origins,
        original_sources=ownership.sources(config['lane'])['original'],serializer_source=source_record(__file__),
        rows=[dump(x)for x in prepared],conditions=audio_conditions(prepared).detach().cpu(),complete=True,model_or_teacher_calls=0))

def load_cache(pin,*,config):
    packet=read_packet(pin)
    keys={'schema','profile','cache_id','scope','teacher_identity','teacher_origins','original_sources','serializer_source','rows','conditions','complete','model_or_teacher_calls'}
    if(not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=SCHEMA or packet['profile']!=PROFILE
            or packet['complete']is not True or packet['model_or_teacher_calls']!=0
            or packet['scope']not in('completed-native','synthetic-software-only')
            or(config['lane']=='native-nf4'and packet['scope']!='completed-native')
            or packet['teacher_identity']!=teacher_identity(config)
            or packet['original_sources']!=ownership.sources(config['lane'])['original']
            or packet['serializer_source']!=source_record(__file__)):
        raise ValueError('exact completed cache/role/lane/Source required; no migration of another interface or original cohort')
    prepared=[restore(x,'cpu')for x in packet['rows']];validate_prepared(prepared,config)
    origins_valid(packet['teacher_origins'],prepared,config,packet['scope'])
    if not equal(audio_conditions(prepared),packet['conditions']):raise ValueError('original audio_conditions differs')
    identity={key:packet[key]for key in('schema','profile','cache_id','scope','teacher_identity','teacher_origins','original_sources','serializer_source')}
    identity.update(whole_file=dict(path=pin.path,bytes=pin.bytes,sha256=pin.sha256),row_count=len(prepared),
        response_shape=[1,config['width']],game_response_scale=1.0)
    return prepared,packet['conditions'],identity

def to_device(prepared,device):return[restore(dump(item),device)for item in prepared]
