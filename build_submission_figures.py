"""Figures from final holdout. Selection rules saved; never used for tuning."""
import json
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from vision import Segmenter
from build_design_review import GROUP_NAMES
ROOT=Path(__file__).parent;R=ROOT/'outputs/submission/vision';D=ROOT/'data/submission/vision';M=ROOT/'model/submission';FONT='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'

def main():
 torch.set_num_threads(1);result=json.loads((R/'final_comparison.json').read_text());data=np.load(D/'test_512.npz');net=Segmenter();net.load_state_dict(torch.load(M/'refined.pt',weights_only=True));net.eval();chosen=[]
 for group in ['tiny_fragments','bed_similar','unseen_hues']:
  rows=[r for r in result['rows'] if r['group']==group and r['present']]
  # First improvement if present; otherwise first remaining miss; otherwise first positive.
  target=next((r for r in rows if r['baseline128_clear'] and not r['improved512_clear']),None)
  reason='first baseline miss corrected' if target else 'first remaining miss or first positive'
  if target is None:target=next((r for r in rows if r['improved512_clear']),rows[0])
  ix=target['id'];im=data['images'][ix];gt=data['masks'][ix];
  with torch.no_grad():pred=torch.sigmoid(net(torch.from_numpy(im.copy()).permute(2,0,1)[None].float()/255))[0,0].numpy()>=.5
  overlay=im.copy();overlay[pred]=(overlay[pred]*.45+np.array([210,35,25])*.55).astype(np.uint8);truth=im.copy();truth[gt>0]=(truth[gt>0]*.45+np.array([20,170,80])*.55).astype(np.uint8)
  canvas=Image.new('RGB',(1280,720),'white');draw=ImageDraw.Draw(canvas);head=ImageFont.truetype(FONT,31);body=ImageFont.truetype(FONT,24)
  draw.text((35,24),GROUP_NAMES[group]+f' / 未使用評価画像 {ix}',font=head,fill='black')
  for j,(a,label) in enumerate([(im,'補正画像'),(truth,'緑：正解領域'),(overlay,'赤：追加学習CNNの推定')]):
   canvas.paste(Image.fromarray(a).resize((390,390)),(35+420*j,130));draw.text((35+420*j,90),label,font=body,fill='black')
   if group=='tiny_fragments' and gt.sum():
    ys,xs=np.where(gt>0);x=int(np.clip(xs.mean()-24,0,464));y=int(np.clip(ys.mean()-24,0,464))
    draw.rectangle((35+420*j+x*390/512,130+y*390/512,35+420*j+(x+48)*390/512,130+(y+48)*390/512),outline='black',width=2)
    canvas.paste(Image.fromarray(a[y:y+48,x:x+48]).resize((144,144),Image.Resampling.NEAREST),(275+420*j,155));draw.rectangle((275+420*j,155,419+420*j,299),outline='black',width=2);draw.rectangle((277+420*j,157,417+420*j,184),fill='white');draw.text((282+420*j,157),'拡大',font=ImageFont.truetype(FONT,20),fill='black')
  status=lambda v:'見逃し' if v else '残留物を検出（停止）'
  draw.text((35,555),'基準CNN：'+status(target['baseline128_clear'])+' / 追加学習後：'+status(target['improved512_clear']),font=body,fill='black')
  draw.text((35,605),'合成画像。正解マスクは採点・表示だけに使用。',font=body,fill='black')
  canvas.save(R/f'example_{group}.png');chosen.append({'id':ix,'group':group,'rule':reason})
 (R/'figure_selection.json').write_text(json.dumps(chosen,indent=2)+'\n')
if __name__=='__main__':main()
