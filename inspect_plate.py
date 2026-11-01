"""RGB-only residual inspection. No printer commands or simulator state inputs.
Input: RGB array, camera intrinsics and distortion. Output: validity, residual
count and clear decision. The decision means plate inspection only, not successful
bin collection, safe temperature, release from adhesion, or permission to print.
"""
import json
from pathlib import Path
import numpy as np
import torch
from vision import Segmenter
from camera_geometry import estimate_plane,rectify
ROOT=Path(__file__).parent

class PlateInspector:
    def __init__(self,model_dir=None):
        folder=Path(model_dir) if model_dir else ROOT/'model/submission'
        self.config=json.loads((folder/'final_selection.json').read_text());self.model=Segmenter()
        self.model.load_state_dict(torch.load(folder/'refined.pt',weights_only=True,map_location='cpu'));self.model.eval()
        self.minimum=self.config['configs']['refined512']['minimum_pixels']
    def inspect(self,rgb,K,distortion):
        location=estimate_plane(rgb,K,distortion)
        if not location['ok']:return {'valid':False,'clear':False,'reason':location['reason']}
        crop=rectify(rgb,K,distortion,location,512)
        with torch.no_grad():prob=torch.sigmoid(self.model(torch.from_numpy(crop.copy()).permute(2,0,1)[None].float()/255))[0,0].numpy()
        count=int((prob>=self.config['probability']).sum())
        return {'valid':True,'clear':count<self.minimum,'foreground_pixels':count,'minimum_pixels':self.minimum,'marker_ids':location['ids']}
