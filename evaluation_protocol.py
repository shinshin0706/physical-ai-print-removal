"""Freeze evaluation inputs BEFORE measuring outcomes. No training or retuning."""
from pathlib import Path
import hashlib,json,itertools
from a1_workcell import ROOT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create():
    out=ROOT/'data/evaluation';out.mkdir(exist_ok=True)
    scene=(ROOT/'outputs/proposed_workcell.xml').read_bytes();(out/'scene.xml').write_bytes(scene)
    path=(ROOT/'outputs/cad_motion_validation.json').read_bytes();(out/'trajectory.json').write_bytes(path)
    baseline=json.loads((ROOT/'outputs/webcam_vision_baseline.json').read_text())['config']
    protocol={'purpose':'Frozen synthetic stress tests of existing motion and image classifiers; not real-device success rates',
      'model_sha256':sha(ROOT/'model/webcam/segmenter.pt'),'model_config_sha256':sha(ROOT/'model/webcam/config.json'),
      'scene_sha256':sha(out/'scene.xml'),'trajectory_sha256':sha(out/'trajectory.json'),
      'baseline_config':baseline,'training_or_threshold_retuning_allowed':False,
      'vision':{'seed':18701,'groups':['reference','bed_similar','achromatic','unseen_hues','tiny_fragments','directional_shadow','low_exposure'],
                'samples_per_group':40,'present_per_group':20,'resolution':[1280,720],'crop_resolution':[128,128],
                'empty_and_present_pairs':True,'tiny_full_side_mm':[1.5,5.0],
                'failure_policy':'fewer than 3 markers or invalid homography => inspection rejected, clear=False; counted separately',
                'metrics':['false_clear/present','false_block/empty','localization_rejection','IoU where visible crop mask exists']},
      'motion':{'masses_kg':[.004,.008,.020],'surface_friction':[.1,.3,.6],
                'along_offsets_m':[-.015,0,.015],'cross_offsets_m':[-.020,0,.020],
                'dimensions_m':[.03,.03,.02],'total_cases':81,'trajectory_adjustment':False,
                'contact_pairs':'explicit part-to-plate and part-to-chute friction; other interfaces fixed',
                'material':'mass range below 22.3g fully dense PLA proxy; inertia recomputed from box mass',
                'adhesion':False,'torque_cap_Nm':.980665,'success':'all part corners in bin for last 0.5s, speed<0.01m/s, angular speed<0.1rad/s, max joint error<3deg, no robot/static contact >0.01N',
                'bin_interior_m':{'x':[-.204,.204],'y':[-.779,-.491],'z':[.557,.649]},'bin_tolerance_m':.001,
                'forces_used_for_scoring_only':True,'camera_used_for_motion_control':False,
                'video_selection':'nominal-like mass8g friction0.3 centered; then first failed grid case by ID; show outcome without tuning'} }
    dest=out/'protocol.json'
    if dest.exists() and json.loads(dest.read_text())!=protocol:raise RuntimeError('Existing protocol differs: do not silently redefine evaluation')
    dest.write_text(json.dumps(protocol,indent=2)+'\n');print(dest,sha(dest))
    return protocol


if __name__=='__main__':create()
