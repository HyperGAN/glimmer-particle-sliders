"""Standalone relocation checks; no umbrella checkout or historical imports."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import glimmer_modern


def provenance():
    return json.loads((Path(__file__).resolve().parents[1]/'docs/extraction-provenance.json').read_text())


def test_installed_runtime_files_retain_exact_reviewed_source_bytes():
    root=Path(glimmer_modern.__file__).resolve().parent
    migration=json.loads((Path(__file__).resolve().parents[1]/'docs/core-migration.json').read_text())
    lr_migration=json.loads((Path(__file__).resolve().parents[1]/'docs/lr-migration.json').read_text())
    records=[item for item in provenance()['extracted_files']
             if item['destination_path'].startswith('src/glimmer_modern/')]
    assert len(records)==8
    assert 'site-packages' in str(root)
    for record in records:
        value=(root/Path(record['destination_path']).name).read_bytes()
        expected=lr_migration['product_sources'].get(record['destination_path'], migration['product_sources'][record['destination_path']])
        assert hashlib.sha256(value).hexdigest()==expected
        if record['destination_path'] not in migration['changed_runtime_files'] and record['destination_path'] not in lr_migration['changed_runtime_files']:
            assert hashlib.sha256(value).hexdigest()==record['source_sha256']==record['destination_sha256']


def test_independent_namespace_and_exact_entrypoint_registration():
    assert importlib.util.find_spec('glimmer_slider') is None
    assert importlib.metadata.version('meta-glimmer-modern-response')=='0.1.1'
    for group in ('particle_sliders.response_backends','particle_sliders.inference_backends'):
        entries=list(importlib.metadata.entry_points(group=group,name='glimmer-modern'))
        assert len(entries)==1
        assert entries[0].value.startswith('glimmer_modern.lifecycle:')


def test_source_lock_relocation_and_cleared_core_pin():
    from glimmer_modern.provenance import verify_sources
    from glimmer_modern import PROTOCOL, INITIALIZATION_POLICY
    _,identity=verify_sources()
    assert identity['core_commit']=='ddb41423bcefc1fe76cd94d0f9b367353b3ea137'
    assert PROTOCOL=='glimmer-modern-final-hidden-hg-hndl-v4-lr-cpu'
    assert INITIALIZATION_POLICY=='cpu-sha256-component-streams-v1'


def test_lineage_has_no_history_mirror_or_historical_runtime_copy():
    value=provenance()
    assert value['source_repository']=='mikkel/meta-glimmer-particle-slider'
    assert value['source_commit']=='3f0cdc4d17ebc6fa2df36d65b001ecfaf436c577'
    assert value['destination_repository']=='HyperGAN/glimmer-particle-sliders'
    assert len(value['extracted_files'])==12
    assert all(item['source_path'].startswith('packages/glimmer-modern-response/')
               for item in value['extracted_files'])
    historical={item['source_path'] for item in value['deferred_historical']}
    assert {'src/glimmer_slider/upstream.py','src/glimmer_slider/training.py','pyproject.toml'}<=historical
    assert not (Path(__file__).resolve().parents[1]/'src/glimmer_slider').exists()
