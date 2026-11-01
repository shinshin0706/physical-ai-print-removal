"""Refine the CNN with native-scale 128px patches from 512px corrected images.
Development is based on validation only. Pilot test outcomes are not consulted.
Final thresholds: minimize false clears subject to <=10% empty false blocks
(including localization rejection); if infeasible minimize false blocks first.
Final test seed31701 is generated only after all model/threshold choices freeze.
"""
import json,argparse,shutil
from pathlib import Path
import numpy as np,torch,cv2
from torch import nn
from submission_scene import P,O,ROOT
from vision import Segmenter
from improve_vision import GROUPS,metrics,rule
from evaluation_protocol import sha
D=P/'vision';R=O/'vision';M=ROOT/'model/submission'

def make_patches():
    src=np.load(D/'patchtrain_512.npz');meta=json.loads((D/'patchtrain.json').read_text());rng=np.random.default_rng(35701)
    patches=[];labels=[];which=[]
    for row,im,gt,valid in zip(meta['rows'],src['images'],src['masks'],src['valid']):
        if not valid:continue
        # Every positive receives one object-centred patch and two random patches.
        starts=[];ys,xs=np.where(gt>0)
        if len(xs):starts.append((int(np.clip(xs.mean()+rng.integers(-28,29)-64,0,384)),int(np.clip(ys.mean()+rng.integers(-28,29)-64,0,384))))
        starts += [(int(rng.integers(0,385)),int(rng.integers(0,385))) for _ in range(2)]
        for x,y in starts:patches.append(im[y:y+128,x:x+128]);labels.append(gt[y:y+128,x:x+128]);which.append(row['pair']%5==0)
    patches=np.array(patches);labels=np.array(labels);which=np.array(which)
    for name,mask in [('train',~which),('validation',which)]:np.savez_compressed(D/f'patch_{name}.npz',images=patches[mask],masks=labels[mask])
    (D/'patch_manifest.json').write_text(json.dumps({'source_sha256':sha(D/'patchtrain_512.npz'),'split':'pair index divisible by5 held out; no source frame crosses split','train_patches':int((~which).sum()),'validation_patches':int(which.sum()),'seed':35701},indent=2)+'\n')
    print('patches',len(patches),int((~which).sum()),int(which.sum()),flush=True)

def train():
    torch.set_num_threads(2);torch.manual_seed(36701);torch.use_deterministic_algorithms(True)
    # Start from public-scene baseline; fine-tune all convolution weights.
    net=Segmenter();net.load_state_dict(torch.load(M/'segmenter.pt',weights_only=True,map_location='cpu'))
    def load(name):
        a=np.load(D/f'patch_{name}.npz');return torch.from_numpy(a['images'].copy()).permute(0,3,1,2).float()/255,torch.from_numpy(a['masks'].copy())[:,None].float()
    x,y=load('train');vx,vy=load('validation');opt=torch.optim.Adam(net.parameters(),lr=.001);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.]))
    best=float('inf');history=[]
    for epoch in range(50):
        net.train();trainloss=[]
        for ids in torch.randperm(len(x)).split(16):
            opt.zero_grad();loss=lossfn(net(x[ids]),y[ids]);loss.backward();opt.step();trainloss.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():losses=[float(lossfn(net(b),c)) for b,c in zip(vx.split(16),vy.split(16))]
        value=float(np.mean(losses));history.append({'epoch':epoch+1,'train_loss':float(np.mean(trainloss)),'checkpoint_validation_loss':value})
        if value<best:best=value;torch.save(net.state_dict(),M/'refined.pt')
        if (epoch+1)%5==0:print('refine',epoch+1,value,flush=True)
    (M/'refined_training.json').write_text(json.dumps({'seed':36701,'epochs':50,'learning_rate':.001,'positive_weight':10,'initial_weights_sha256':sha(M/'segmenter.pt'),'weights_sha256':sha(M/'refined.pt'),'patch_manifest_sha256':sha(D/'patch_manifest.json'),'checkpoint_selection':'lowest held-out patch BCE; test untouched','history':history},indent=2)+'\n')

def predict(split):
    torch.set_num_threads(2);a=np.load(D/f'{split}_512.npz');net=Segmenter();net.load_state_dict(torch.load(M/'refined.pt',weights_only=True,map_location='cpu'));net.eval();counts=[]
    with torch.no_grad():
        for first in range(0,len(a['images']),4):
            im=a['images'][first:first+4];prob=torch.sigmoid(net(torch.from_numpy(im.copy()).permute(0,3,1,2).float()/255))[:,0].numpy();counts.extend((prob>=.5).sum((1,2)).tolist())
    return np.array(counts)

def choose(counts,present,valid):
    options=[]
    for minimum in [1,2,4,8,12,20,32,64,128,320,640,1280,2560,5120,16384,65536]:
        r=metrics((counts<minimum)&valid,present,valid);feasible=r['false_block']<=.10*r['empty']
        key=(not feasible,r['false_clear'] if feasible else r['false_block'],r['false_block'] if feasible else r['false_clear'],-minimum)
        options.append((key,minimum,r,feasible))
    _,minimum,r,feasible=min(options,key=lambda a:a[0]);return {'minimum_pixels':minimum,'validation':r,'false_block_budget_met':feasible}

def select():
    c=np.load(R/'validation_counts.npz');fresh=predict('validation');methods={name:choose(c[name],c['present'],c['valid']) for name in ['cnn128','cnn512','rule512']};methods['refined512']=choose(fresh,c['present'],c['valid'])
    np.savez_compressed(R/'refined_validation_counts.npz',counts=fresh)
    plan={'test_seed':31701,'test_count':560,'groups':GROUPS,'scene_sha256':sha(P/'scene.xml'),'trajectory_sha256':sha(P/'trajectory.json'),'original_weights_sha256':sha(M/'segmenter.pt'),'refined_weights_sha256':sha(M/'refined.pt'),'validation_sha256':sha(D/'validation.json'),'objective':'min false_clear subject to false_block<=10% of empty; if infeasible min false_block then false_clear; tie larger threshold','threshold_candidates':[1,2,4,8,12,20,32,64,128,320,640,1280,2560,5120,16384,65536],'configs':methods,'probability':.5,'baseline128_minimum':20,'area_equivalent512_minimum':320,'pilot':'Unconstrained weighted objective produced an always-stop detector on validation; rejected before looking at pilot test outcomes.'}
    (M/'final_selection.json').write_text(json.dumps(plan,indent=2)+'\n');print(plan,flush=True)

def evaluate():
    from improve_vision import count_predictions
    config=json.loads((M/'final_selection.json').read_text());meta=json.loads((D/'test.json').read_text());assert meta['seed']==config['test_seed']
    assert sha(M/'segmenter.pt')==config['original_weights_sha256'];assert sha(M/'refined.pt')==config['refined_weights_sha256']
    c=count_predictions('test');c['refined512']=predict('test');truth=c['present'];valid=c['valid'];rows=meta['rows'];groups=np.array([r['group'] for r in rows]);methods={}
    variants={'baseline128':('cnn128',20),'threshold128':('cnn128',config['configs']['cnn128']['minimum_pixels']),'resolution512':('cnn512',320),'tuned512':('cnn512',config['configs']['cnn512']['minimum_pixels']),'improved512':('refined512',config['configs']['refined512']['minimum_pixels']),'rule512':('rule512',config['configs']['rule512']['minimum_pixels'])}
    for name,(key,minimum) in variants.items():
        clear=(c[key]<minimum)&valid;methods[name]={'minimum_pixels':minimum,'overall':metrics(clear,truth,valid),'by_group':{g:metrics(clear[groups==g],truth[groups==g],valid[groups==g]) for g in GROUPS}}
        for r,cl in zip(rows,clear):r[name+'_clear']=bool(cl)
    np.savez_compressed(R/'final_test_counts.npz',**c)
    result={'scope':'frozen final holdout on independent public scene','selection_sha256':sha(M/'final_selection.json'),'test_manifest_sha256':sha(D/'test.json'),'methods':methods,'rows':rows,'script_sha256':sha(__file__)}
    (R/'final_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v['overall'] for k,v in methods.items()},flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('cmd',choices=['patches','train','select','evaluate']);cmd=a.parse_args().cmd
    {'patches':make_patches,'train':train,'select':select,'evaluate':evaluate}[cmd]()
