"""Public-scene controlled resolution ablation, with validation-only selection.
No private CAD, geometry labels at inference, or previous test examples are used.
Commands: generate-train, train, generate-validation, select, generate-test, evaluate.
Old pipeline is reproduced as an architectural baseline using newly trained public
weights, NOT called the identical previous model. Four variants share these weights.
"""
import argparse,json,hashlib,time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,cv2,mujoco,torch
from PIL import Image
from submission_scene import ROOT,P,O,PLATE_Z,BED_Y,vec
from camera_geometry import estimate_plane,rectify
from demo_camera import intrinsics
from vision import Segmenter,train,PALETTE
from compare_webcam_vision import rule
from evaluation_protocol import sha
GROUPS=['reference','bed_similar','achromatic','unseen_hues','tiny_fragments','directional_shadow','low_exposure']
D=P/'vision';M=ROOT/'model/submission';R=O/'vision'

def generate(split):
    D.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if split=='test' and not (M/'final_selection.json').exists():raise RuntimeError('Select and freeze on validation before generating test')
    seed={'train':24701,'validation':28701,'test':31701,'patchtrain':34701}[split];pairs={'train':120,'validation':20,'test':40,'patchtrain':40}[split]
    root=ET.parse(P/'scene.xml').getroot();world=root.find('worldbody')
    ET.SubElement(world,'geom',name='sample',type='box',size='.02 .02 .01',pos='0 0 -1',rgba='.9 .3 .05 1')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m);d.qpos[:5]=json.loads((P/'trajectory.json').read_text())['samples'][-1]['q_rad'];d.qpos[5]=.2
    gid=m.geom('sample').id;cid=m.camera('webcam').id;K=intrinsics(m,1280,720);cam=m.cam_pos[cid].copy();lp=m.light_pos.copy();ld=m.light_dir.copy();ldiff=m.light_diffuse.copy();ha=m.vis.headlight.ambient.copy();hd=m.vis.headlight.diffuse.copy()
    rng=np.random.default_rng(seed);rows=[];ims128=[];ims512=[];gt128=[];gt512=[]
    with mujoco.Renderer(m,720,1280) as renderer:
        for group in (['reference'] if split=='train' else (['reference','tiny_fragments','directional_shadow','low_exposure'] if split=='patchtrain' else GROUPS)):
            for pair in range(pairs):
                half=np.r_[rng.uniform(.008,.025,2),rng.uniform(.003,.020)]
                if split=='train' and pair%2:half=np.r_[rng.uniform(.004,.012,2),rng.uniform(.001,.003)]
                if group=='tiny_fragments':half=np.r_[rng.uniform(.00075,.0025,2),rng.uniform(.0005,.0015)]
                color=PALETTE[pair%4]*rng.uniform(.85,1.1)
                if group=='bed_similar':color=np.clip([.55,.43,.19]+rng.uniform(-.018,.018,3),0,1)
                if group=='achromatic':color=np.full(3,[.04,.4,.9][pair%3])
                if group=='unseen_hues':color=np.array([[.08,.75,.8],[.60,.12,.8],[.95,.35,.60],[.9,.9,.08]])[pair%4]
                xy=rng.uniform(-.085,.085,2);angle=rng.uniform(-np.pi,np.pi);delta=rng.uniform([-.025,-.015,-.025],[.025,.015,.025]);illum=float(rng.uniform(.75,1.15));gain=float(rng.uniform(.25,.5)) if group=='low_exposure' else 1.
                m.cam_pos[cid]=cam+delta;m.light_pos[:]=lp;m.light_dir[:]=ld;m.light_diffuse[:]=ldiff*illum;m.vis.headlight.ambient[:]=ha;m.vis.headlight.diffuse[:]=hd
                if group=='directional_shadow':
                    m.light_pos[0]=[rng.uniform(-.65,.65),rng.uniform(-.15,.4),rng.uniform(1.25,1.55)];direction=np.array([0,BED_Y,PLATE_Z])-m.light_pos[0];m.light_dir[0]=direction/np.linalg.norm(direction);m.vis.headlight.ambient[:]=.08;m.vis.headlight.diffuse[:]=.10
                m.geom_size[gid]=half;m.geom_quat[gid]=[np.cos(angle/2),0,0,np.sin(angle/2)];m.geom_rgba[gid,:3]=color
                noise=rng.normal(0,1.4,(720,1280,3)).astype(np.float32) # shared raw noise in each pair
                for present in [False,True]:
                    m.geom_pos[gid]=[xy[0],xy[1]+BED_Y,PLATE_Z+half[2] if present else -1];mujoco.mj_forward(m,d)
                    renderer.update_scene(d,camera='webcam');raw=np.clip(renderer.render().astype(np.float32)*gain+noise,0,255).astype(np.uint8)
                    loc=estimate_plane(raw,K,np.zeros(5));renderer.enable_segmentation_rendering();renderer.update_scene(d,camera='webcam');seg=renderer.render().copy();renderer.disable_segmentation_rendering();mask=((seg[:,:,0]==gid)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))).astype(np.uint8)
                    row={'id':len(rows),'split':split,'group':group,'pair':pair,'present':present,'valid':bool(loc['ok']),'dimensions_m':(2*half).tolist(),'rgb':color.tolist(),'xy_plate_m':xy.tolist(),'raw_sha256':hashlib.sha256(raw.tobytes()).hexdigest()}
                    for size,images,masks in [(128,ims128,gt128),(512,ims512,gt512)]:
                        crop=np.zeros((size,size,3),np.uint8);gt=np.zeros((size,size),np.uint8)
                        if loc['ok']:
                            crop=rectify(raw,K,np.zeros(5),loc,size)
                            T=np.array([[size/.256,0,size/2],[0,-size/.256,size/2],[0,0,1.]])@np.array(loc['H_pixel_to_plane']);gt=cv2.warpPerspective(mask,T,(size,size),flags=cv2.INTER_NEAREST)
                        images.append(crop);masks.append(gt);row[f'gt_pixels_{size}']=int(gt.sum())
                    if split=='test':
                        (R/'test_frames').mkdir(exist_ok=True);Image.fromarray(raw).save(R/'test_frames'/f'{len(rows):04d}.png')
                    rows.append(row)
            print('generated',split,group,len(rows),flush=True)
    for size,images,masks in [(128,ims128,gt128),(512,ims512,gt512)]:
        np.savez_compressed(D/f'{split}_{size}.npz',images=np.array(images),masks=np.array(masks),present=np.array([r['present'] for r in rows]),valid=np.array([r['valid'] for r in rows]))
    info={'seed':seed,'rows':rows,'scene_sha256':sha(P/'scene.xml'),'script_sha256':sha(__file__),'inspection_q_rad':d.qpos[:5].tolist(),'trajectory_sha256':sha(P/'trajectory.json'),'files':{str(n):sha(D/f'{split}_{n}.npz') for n in [128,512]}}
    (D/f'{split}.json').write_text(json.dumps(info,indent=2)+'\n')

def fit():
    # Model selection uses a disjoint 60-example subset of training-generation
    # pairs; the stress validation split remains for decision threshold selection.
    source=np.load(D/'train_128.npz');folder=D/'fit';folder.mkdir(exist_ok=True)
    # 240 total =>180 train,60 checkpoint-validation. Keep entire empty/present pairs together.
    for split,sl in [('train',slice(0,180)),('validation',slice(180,240))]:np.savez_compressed(folder/f'{split}.npz',images=source['images'][sl],masks=source['masks'][sl])
    (folder/'manifest.json').write_text(json.dumps({'source_sha256':sha(D/'train_128.npz'),'split':'first90 pairs train; last30 pairs checkpoint-validation','seed':24701}))
    torch.set_num_threads(2);train(folder,M,epochs=120)

def count_predictions(split):
    torch.set_num_threads(2);net=Segmenter();net.load_state_dict(torch.load(M/'segmenter.pt',weights_only=True,map_location='cpu'));net.eval();counts={}
    for size in [128,512]:
        data=np.load(D/f'{split}_{size}.npz');im=data['images'];start=time.monotonic();pixel=np.zeros(len(im),int)
        with torch.no_grad():
            for first in range(0,len(im),4):
                p=torch.sigmoid(net(torch.from_numpy(im[first:first+4].copy()).permute(0,3,1,2).float()/255))[:,0].numpy()
                pixel[first:first+4]=(p>=.5).sum((1,2))
        counts[f'cnn{size}']=pixel;print('inference',split,size,time.monotonic()-start,flush=True)
        if size==512:counts['rule512']=rule(im,.015,.15).sum((1,2))
    counts.update(present=data['present'],valid=data['valid']);return counts

def metrics(clear,present,valid):
    return {'count':len(clear),'present':int(present.sum()),'empty':int((~present).sum()),'false_clear':int((clear&present).sum()),'false_block':int((~clear&~present).sum()),'rejected':int((~valid).sum())}

def select():
    if (D/'test.json').exists():raise RuntimeError('Do not reselect after test exists')
    counts=count_predictions('validation');truth=counts['present'];valid=counts['valid'];configs={}
    for name in ['cnn128','cnn512','rule512']:
        best=None
        for minimum in [1,2,4,8,12,20,32,64,128,320]:
            clear=(counts[name]<minimum)&valid;r=metrics(clear,truth,valid);key=(20*r['false_clear']+r['false_block'],-minimum)
            if best is None or key<best[0]:best=(key,minimum,r)
        configs[name]={'minimum_pixels':best[1],'validation':best[2]}
    out={'scene_sha256':sha(P/'scene.xml'),'weights_sha256':sha(M/'segmenter.pt'),'validation_sha256':sha(D/'validation.json'),'test_seed':29701,'test_count':560,'candidates':[1,2,4,8,12,20,32,64,128,320],'objective':'minimize 20*false_clear+false_block; tie prefer larger minimum','probability':.5,'rule_hue_gap':.015,'rule_saturation_min':.15,'configs':configs,'baseline128_minimum':20,'ablation512_area_equivalent_minimum':320,'no_test_tuning':True}
    (M/'selection.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(R/'validation_counts.npz',**counts);print(out)

def evaluate():
    config=json.loads((M/'selection.json').read_text());assert sha(M/'segmenter.pt')==config['weights_sha256'];assert sha(P/'scene.xml')==config['scene_sha256']
    counts=count_predictions('test');truth=counts['present'];valid=counts['valid'];meta=json.loads((D/'test.json').read_text());rows=meta['rows'];groups=np.array([r['group'] for r in rows]);methods={}
    variants={'baseline128':('cnn128',20),'threshold128':('cnn128',config['configs']['cnn128']['minimum_pixels']),'resolution512':('cnn512',320),'improved512':('cnn512',config['configs']['cnn512']['minimum_pixels']),'rule512':('rule512',config['configs']['rule512']['minimum_pixels'])}
    for name,(source,minimum) in variants.items():
        clear=(counts[source]<minimum)&valid;methods[name]={'minimum_pixels':minimum,'overall':metrics(clear,truth,valid),'by_group':{g:metrics(clear[groups==g],truth[groups==g],valid[groups==g]) for g in GROUPS}}
        for r,c in zip(rows,clear):r[name+'_clear']=bool(c)
    result={'scope':'new independent primitive scene and newly trained shared CNN, not old CAD evaluation','methods':methods,'rows':rows,'selection_sha256':sha(M/'selection.json'),'test_manifest_sha256':sha(D/'test.json'),'model_sha256':sha(M/'segmenter.pt'),'script_sha256':sha(__file__)}
    (R/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(R/'test_counts.npz',**counts);print({k:v['overall'] for k,v in methods.items()})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['generate-train','generate-validation','generate-test','generate-patchtrain','train','select','evaluate']);cmd=a.parse_args().command
    if cmd.startswith('generate-'):generate(cmd.split('-')[1])
    elif cmd=='train':fit()
    elif cmd=='select':select()
    else:evaluate()
