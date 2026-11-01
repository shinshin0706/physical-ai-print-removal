"""Run the same fixed 81-condition grid on the independently made public scene.
Keeps old CAD results separate. This is a primitive-scene result, not an A1
geometry certificate. No adaptation to per-case object position or friction.
"""
import json,argparse
import evaluate_motion as engine
from submission_scene import P,O,PLATE_Z
from evaluation_protocol import sha

def configure():
    engine.INPUT=P;engine.OUT=O/'motion';engine.OUT.mkdir(exist_ok=True);engine.PLATE_Z=PLATE_Z

def main(videos=False):
    configure()
    if not videos:
        p={'scene_sha256':sha(P/'scene.xml'),'trajectory_sha256':sha(P/'trajectory.json'),'scope':'independent primitive scene, no MakerWorld CAD',
           'motion':{'masses_kg':[.004,.008,.020],'surface_friction':[.1,.3,.6],'along_offsets_m':[-.015,0,.015],'cross_offsets_m':[-.020,0,.020],'total_cases':81}}
        target=P/'protocol.json'
        if target.exists() and json.loads(target.read_text())!=p:raise RuntimeError('Frozen submission motion protocol changed')
        target.write_text(json.dumps(p,indent=2)+'\n')
    engine.main(videos)
    if videos:
        result=json.loads((engine.OUT/'motion_results.json').read_text());success=next((r for r in result['results'] if r['success']),None)
        if success and success['id'] not in result['video_case_ids']:
            case={k:success[k] for k in ['id','mass_kg','friction','along_offset_m','cross_offset_m']};r=engine.run(case,engine.OUT/f'{success["id"]}.mp4')
            import numpy as np
            assert r['success'];np.testing.assert_allclose(r['final_part_xyz'],success['final_part_xyz'],atol=1e-9)
            (engine.OUT/'illustration.json').write_text(json.dumps({'success_id':success['id'],'selection':'first success after evaluation; predeclared cases retained'}))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--videos',action='store_true');main(a.parse_args().videos)
