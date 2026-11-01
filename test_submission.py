"""Tests for the redistributable experiment; no private CAD required."""
import json
import pytest
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from submission_scene import P,O,build
from improve_vision import metrics
from evaluation_protocol import sha
from evaluate_motion import CORNERS,inside_bin

def test_public_scene_does_not_depend_on_restricted_assets():
    root,_=build();xml=ET.tostring(root,encoding='unicode')
    assert 'local_assets' not in xml and 'a1_00' not in xml
    for mesh in root.findall('asset/mesh'):
        path=mesh.get('file');assert '/vendor/robotstudio_so101/' in path or path.endswith('/cad/spatula.stl')
    assert len(root.findall('worldbody/geom'))>100

def test_scoring_keeps_rejected_and_zero_pixel_objects():
    present=np.array([True,False,True]);valid=np.array([False,False,True]);clear=np.array([False,False,True])
    r=metrics(clear,present,valid);assert r['false_clear']==1 and r['false_block']==1 and r['rejected']==2

def test_full_object_must_fit_bin():
    assert inside_bin(np.array([[.20,-.63,.60]]))
    assert not inside_bin(CORNERS+[.20,-.63,.60])

@pytest.mark.skipif(not (P/"vision/train_128.npz").exists(), reason="Full generated datasets are not bundled; regenerate to check byte-level provenance")
def test_new_splits_do_not_overlap_and_model_is_frozen():
    metas={s:json.loads((P/f'vision/{s}.json').read_text()) for s in ['train','validation','test']}
    raw={s:{r['raw_sha256'] for r in m['rows']} for s,m in metas.items()}
    assert not(raw['train']&raw['validation'] or raw['train']&raw['test'] or raw['validation']&raw['test'])
    assert len(metas['test']['rows'])==560
    selection=json.loads((P.parent.parent/'model/submission/final_selection.json').read_text())
    assert selection['original_weights_sha256']==sha(P.parent.parent/'model/submission/segmenter.pt')
    assert selection['scene_sha256']==sha(P/'scene.xml')
    assert selection['validation_sha256']==sha(P/'vision/validation.json')
    for split,meta in metas.items():
        assert meta['trajectory_sha256']==sha(P/'trajectory.json')
        for size in [128,512]:assert meta['files'][str(size)]==sha(P/f'vision/{split}_{size}.npz')

def test_classification_can_be_recomputed_from_saved_counts():
    result=json.loads((O/'vision/final_comparison.json').read_text());counts=np.load(O/'vision/final_test_counts.npz')
    sources={'baseline128':'cnn128','threshold128':'cnn128','resolution512':'cnn512','tuned512':'cnn512','improved512':'refined512','rule512':'rule512'}
    for name,method in result['methods'].items():
        clear=(counts[sources[name]]<method['minimum_pixels'])&counts['valid']
        assert metrics(clear,counts['present'],counts['valid'])==method['overall']

def test_inspector_rejects_unlocalized_frame():
    from inspect_plate import PlateInspector
    result=PlateInspector().inspect(np.full((720,1280,3),128,np.uint8),np.array([[800,0,640],[0,800,360],[0,0,1.]]),np.zeros(5))
    assert result['valid'] is False and result['clear'] is False

def test_rgb_only_inspection_matches_frozen_evaluation():
    from inspect_plate import PlateInspector
    from PIL import Image
    import torch
    torch.set_num_threads(1)
    root=Path(__file__).parent;example=json.loads((root/'examples/inspection.json').read_text());inspector=PlateInspector()
    for row in example['rows']:
        answer=inspector.inspect(np.array(Image.open(root/'examples'/row['file'])),np.array(example['K']),np.array(example['distortion']))
        assert answer['valid']==row['valid'] and answer['clear']==row['clear']
        if row['valid']:assert answer['foreground_pixels']==row['foreground_pixels']
