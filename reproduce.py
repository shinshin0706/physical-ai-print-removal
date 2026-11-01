"""Reproduce the public experiment without obtaining the restricted printer CAD.
Run with nice -n 15. Modes: scores (saved decisions), vision (fresh holdout with
provided frozen weights), full (regenerate, train, select, then evaluate), motion.
Rendering tested on macOS; Linux headless hosts may need MUJOCO_GL=egl.
"""
import argparse,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).parent

def run(*args):
 subprocess.run([sys.executable,*args],cwd=ROOT,check=True,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'})

def main(mode):
 if mode=='scores':run('-m','pytest','-q','test_submission.py');return
 run('submission_scene.py')
 if mode=='motion':run('submission_motion.py');return
 if mode=='full':
  run('improve_vision.py','generate-train');run('improve_vision.py','train');run('improve_vision.py','generate-validation')
  run('-c',"from improve_vision import count_predictions,R; import numpy as np; np.savez_compressed(R/'validation_counts.npz',**count_predictions('validation'))")
  run('improve_vision.py','generate-patchtrain');run('refine_detector.py','patches');run('refine_detector.py','train');run('refine_detector.py','select')
 run('improve_vision.py','generate-test');run('refine_detector.py','evaluate')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['scores','vision','motion','full']);main(p.parse_args().mode)
