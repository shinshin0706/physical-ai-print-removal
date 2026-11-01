"""Train a small RGB segmentation CNN on simulator images; evaluate held-out seeds.

The network sees only RGB. Geometry IDs supply offline training/evaluation labels,
never inference inputs. The declared color/lighting/shape domain is synthetic.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from removal import ROOT, BED_TOP, RecoveryExperiment, Scenario

SEEDS = {'train': 1701, 'validation': 2701, 'test': 3701}
COUNTS = {'train': 240, 'validation': 60, 'test': 120}
PALETTE = np.array([[.85,.35,.10], [.80,.12,.12], [.10,.35,.85], [.12,.65,.28]])
torch.set_num_threads(4)


class Segmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3,12,3,padding=1), nn.ReLU(),
                                 nn.Conv2d(12,12,3,padding=1), nn.ReLU(), nn.Conv2d(12,1,1))

    def forward(self, x):
        return self.net(x)


def generate_dataset(output):
    output.mkdir(parents=True, exist_ok=True)
    env = RecoveryExperiment(Scenario('dataset', residue=True))
    renderer = mujoco.Renderer(env.model, 128, 128)
    model, data = env.model, env.data
    part_adr = model.joint('part_joint').qposadr[0]
    fragment = model.geom('residue').id
    fragment_body = model.body('residue').id
    bed = model.geom('build_plate').id
    light_default = model.light_diffuse.copy()
    manifests = {}
    try:
        for split, seed in SEEDS.items():
            rng = np.random.default_rng(seed)
            images, masks, entries = [], [], []
            for i in range(COUNTS[split]):
                mode = ('empty', 'object', 'fragment')[i % 3]
                color = np.clip(PALETTE[rng.integers(len(PALETTE))] * rng.uniform(.8,1.15), 0, 1)
                size = np.r_[rng.uniform(.008,.042,2), rng.uniform(.005,.026)]
                xy = rng.uniform(-.075,.075,2)
                angle = rng.uniform(-np.pi,np.pi)
                data.qpos[part_adr:part_adr+3] = [*xy, BED_TOP+size[2]+.0001] if mode == 'object' else [0,0,-1]
                data.qpos[part_adr+3:part_adr+7] = [np.cos(angle/2),0,0,np.sin(angle/2)]
                model.geom_size[env.part_geom] = size
                model.geom_rgba[env.part_geom,:3] = color
                fsize = np.r_[rng.uniform(.004,.016,2), rng.uniform(.001,.003)]
                model.geom_size[fragment] = fsize
                model.body_pos[fragment_body] = [*rng.uniform(-.10,.10,2),BED_TOP+fsize[2]] if mode == 'fragment' else [0,0,-1]
                model.geom_rgba[fragment,:3] = color
                model.geom_rgba[bed,:3] = np.array([.66,.53,.26]) * rng.uniform(.83,1.08)
                model.light_diffuse[:] = light_default * rng.uniform(.65,1.25)
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera='inspection')
                rgb = renderer.render().copy()
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data,camera='inspection')
                seg = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                mask = np.isin(seg[:,:,0],[env.part_geom,fragment]) & (seg[:,:,1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                if mode != 'empty' and mask.sum() == 0:
                    raise RuntimeError('Nonempty sample has no visible foreground')
                # Sensor-like RGB noise; labels stay unchanged. No test augmentation tuning.
                rgb = np.clip(rgb.astype(float)+rng.normal(0,1.4,rgb.shape),0,255).astype(np.uint8)
                images.append(rgb); masks.append(mask.astype(np.uint8))
                entries.append({'index':i,'mode':mode,'foreground_pixels':int(mask.sum()),'rgb':color.tolist()})
            path = output / f'{split}.npz'
            np.savez_compressed(path, images=np.array(images), masks=np.array(masks))
            manifests[split] = {'seed':seed,'count':len(images),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'samples':entries}
            print(f'generated {split}: {len(images)}',flush=True)
    finally:
        renderer.close(); env.close()
    (output/'manifest.json').write_text(json.dumps(manifests,indent=2)+'\n')


def tensors(path):
    d=np.load(path)
    x=torch.from_numpy(d['images'].copy()).permute(0,3,1,2).float()/255
    y=torch.from_numpy(d['masks'].copy()).unsqueeze(1).float()
    return x,y


def predictions(model,x):
    with torch.no_grad():
        return torch.cat([torch.sigmoid(model(batch)) for batch in x.split(20)])


def inspection_roi():
    # Fixed camera calibration: z=1.35 m, plate z=.75 m, vertical FOV 33 deg.
    metres_per_pixel=2*(1.35-BED_TOP)*np.tan(np.deg2rad(33)/2)/128
    coordinate=(np.arange(128)+.5-64)*metres_per_pixel
    return (np.abs(coordinate[:,None])<.128) & (np.abs(coordinate[None,:])<.128)


def summarize(prob,y,min_pixels=4,use_roi=True):
    roi=torch.from_numpy(inspection_roi() if use_roi else np.ones((128,128),dtype=bool))[None,None]
    pred=(prob >= .5) & roi
    target=y.bool() & roi
    intersection=(pred & target).sum(dim=(1,2,3)).numpy()
    union=(pred | target).sum(dim=(1,2,3)).numpy()
    nonempty=target.sum(dim=(1,2,3)).numpy()>0
    occupied=pred.sum(dim=(1,2,3)).numpy()>=min_pixels
    return {'count':len(y),'occupied_count':int(nonempty.sum()),'empty_count':int((~nonempty).sum()),
            'mean_iou_nonempty':float(np.mean(intersection[nonempty]/np.maximum(union[nonempty],1))),
            'false_clear_count':int((nonempty & ~occupied).sum()),
            'false_block_count':int((~nonempty & occupied).sum()),
            'accuracy':float(np.mean(occupied==nonempty))}


def train(data_dir,model_dir,epochs=120):
    torch.manual_seed(4701)
    torch.use_deterministic_algorithms(True)
    model_dir.mkdir(parents=True,exist_ok=True)
    x,y=tensors(data_dir/'train.npz'); vx,vy=tensors(data_dir/'validation.npz')
    model=Segmenter()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([40.]))
    history=[]; best=-1
    for epoch in range(epochs):
        model.train(); losses=[]
        for ix in torch.randperm(len(x)).split(16):
            optimizer.zero_grad(); loss=lossfn(model(x[ix]),y[ix]); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval(); prob=predictions(model,vx)
        scores=summarize(prob,vy,use_roi=False)
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),**scores})
        if scores['mean_iou_nonempty']>best:
            best=scores['mean_iou_nonempty']
            torch.save(model.state_dict(),model_dir/'segmenter.pt')
        if (epoch+1)%5==0:
            print(f'epoch {epoch+1}: loss {np.mean(losses):.4f}, validation IoU {scores["mean_iou_nonempty"]:.3f}',flush=True)
    model.load_state_dict(torch.load(model_dir/'segmenter.pt',weights_only=True))
    prob=predictions(model,vx)
    choices=[]
    for min_pixels in [1,2,4,8,12,20]:
        score=summarize(prob,vy,min_pixels,use_roi=False)
        choices.append((20*score['false_clear_count']+score['false_block_count'],min_pixels,score))
    _,threshold,score=min(choices,key=lambda z:(z[0],-z[1]))
    config={'architecture':'3x3 Conv(3,12), ReLU, 3x3 Conv(12,12), ReLU, 1x1 Conv(12,1)',
            'parameters':sum(p.numel() for p in model.parameters()),'pixel_probability_threshold':.5,
            'minimum_occupied_pixels':threshold,'validation':score,'seed':4701,'epochs':epochs,
            'selection':'best validation nonempty IoU; occupancy threshold minimizes 20*false_clear + false_block on validation',
            'train_manifest_sha256':hashlib.sha256((data_dir/'manifest.json').read_bytes()).hexdigest(),
            'weights_sha256':hashlib.sha256((model_dir/'segmenter.pt').read_bytes()).hexdigest(),
            'torch_version':torch.__version__,'history':history}
    (model_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    return model,config


class Detector:
    def __init__(self,path=ROOT/'model'):
        self.config=json.loads((path/'config.json').read_text())
        self.model=Segmenter()
        self.model.load_state_dict(torch.load(path/'segmenter.pt',weights_only=True,map_location='cpu'))
        self.model.eval()

    def predict(self,rgb):
        x=torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).unsqueeze(0).float()/255
        with torch.no_grad():
            prob=torch.sigmoid(self.model(x))[0,0].numpy()
        mask=(prob >= self.config['pixel_probability_threshold']) & inspection_roi()
        pixels=int(mask.sum())
        return {'clear':pixels<self.config['minimum_occupied_pixels'],'foreground_pixels':pixels,
                'minimum_occupied_pixels':self.config['minimum_occupied_pixels'],
                'mask':mask,'probability':prob}


def evaluate(data_dir,model_dir,output):
    detector=Detector(model_dir)
    x,y=tensors(data_dir/'test.npz')
    prob=predictions(detector.model,x)
    scores=summarize(prob,y,detector.config['minimum_occupied_pixels'])
    scores.update(split='test',seed=SEEDS['test'],weights_sha256=detector.config['weights_sha256'],
                  dataset_sha256=hashlib.sha256((data_dir/'test.npz').read_bytes()).hexdigest())
    output.mkdir(parents=True,exist_ok=True)
    (output/'vision_test.json').write_text(json.dumps(scores,indent=2)+'\n')
    # A fixed, reproducible set of examples, not picked by their performance.
    from PIL import Image,ImageDraw
    panel=Image.new('RGB',(768,822),'white'); draw=ImageDraw.Draw(panel)
    for row,idx in enumerate([0,1,2]):
        rgb=(x[idx].permute(1,2,0).numpy()*255).astype(np.uint8)
        target=y[idx,0].numpy().astype(bool); pred=prob[idx,0].numpy()>=.5
        for col,(name,im) in enumerate([('RGB',rgb),('Ground truth',np.repeat((target*255).astype(np.uint8)[:,:,None],3,axis=2)),('CNN prediction',np.repeat((pred*255).astype(np.uint8)[:,:,None],3,axis=2))]):
            panel.paste(Image.fromarray(im).resize((256,256)),(col*256,row*274+18))
            draw.text((col*256+4,row*274),f'{idx} {name}',fill='black')
    panel.save(output/'vision_examples.png')
    print(json.dumps(scores,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true'); parser.add_argument('--train',action='store_true')
    parser.add_argument('--evaluate',action='store_true'); parser.add_argument('--epochs',type=int,default=120)
    args=parser.parse_args()
    data_dir=ROOT/'data'; model_dir=ROOT/'model'; output=ROOT/'outputs'
    if args.generate: generate_dataset(data_dir)
    if args.train: train(data_dir,model_dir,args.epochs)
    if args.evaluate: evaluate(data_dir,model_dir,output)


if __name__=='__main__': main()
