"""Same held-out webcam dataset: CNN versus validation-tuned hue rule."""
import hashlib,json
import numpy as np
from matplotlib.colors import rgb_to_hsv
from a1_workcell import ROOT


def rule(images,gap,saturation):
    hsv=rgb_to_hsv(images.astype(float)/255)
    hue=np.median(hsv[:,:,:,0],axis=(1,2));d=abs(hsv[:,:,:,0]-hue[:,None,None]);d=np.minimum(d,1-d)
    return (d>gap)&(hsv[:,:,:,1]>saturation)


def scores(mask,truth,threshold):
    real=truth.sum((1,2))>0;pred=mask.sum((1,2))>=threshold
    return {'count':len(real),'false_clear_count':int((real&~pred).sum()),'false_block_count':int((~real&pred).sum()),'accuracy':float((real==pred).mean())}


def main():
    folder=ROOT/'data/webcam';val=np.load(folder/'validation.npz');best=None
    for gap in [.015,.025,.04,.06,.08]:
        for sat in [.15,.25,.4]:
            mask=rule(val['images'],gap,sat)
            for minimum in [1,2,4,8,12,20]:
                score=scores(mask,val['masks'],minimum);key=(20*score['false_clear_count']+score['false_block_count'],-minimum,gap,sat)
                if best is None or key<best[0]:best=(key,{'hue_gap':gap,'saturation_min':sat,'minimum_pixels':minimum},score)
    _,config,validation=best;test=np.load(folder/'test.npz')
    result={'method':'hue difference rule; threshold chosen using validation only','config':config,'validation':validation,
            'test':scores(rule(test['images'],config['hue_gap'],config['saturation_min']),test['masks'],config['minimum_pixels']),
            'cnn_test':json.loads((ROOT/'outputs/webcam_vision_test.json').read_text()),'test_sha256':hashlib.sha256((folder/'test.npz').read_bytes()).hexdigest()}
    (ROOT/'outputs/webcam_vision_baseline.json').write_text(json.dumps(result,indent=2)+'\n');print(result['test'])


if __name__=='__main__':main()
