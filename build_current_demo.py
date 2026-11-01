"""Public 1–3 minute video: observed successes/failures and frozen holdout results."""
from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).parent;O=ROOT/'outputs/submission';E=O/'motion';FONT='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'

def card(title,lines):
 im=Image.new('RGB',(1280,720),'white');d=ImageDraw.Draw(im);heading=ImageFont.truetype(FONT,36);body=ImageFont.truetype(FONT,27)
 assert d.textbbox((0,0),title,font=heading)[2]<1170,title
 d.text((55,85),title,font=heading,fill='black')
 for n,line in enumerate(lines):
  assert d.textbbox((0,0),line,font=body)[2]<1170,line
  d.text((55,205+n*65),line,font=body,fill='#222222')
 return np.array(im)

def main():
 (ROOT/"submission").mkdir(exist_ok=True)
 m=json.loads((E/'motion_results.json').read_text());v=json.loads((O/'vision/final_comparison.json').read_text());a=v['methods']['baseline128']['overall'];b=v['methods']['improved512']['overall'];r=v['methods']['rule512']['overall'];tiny=v['methods']['improved512']['by_group']['tiny_fragments'];timeline=[];frames=0
 with imageio.get_writer(str(ROOT/'submission/demo.mp4'),fps=15,codec='libx264',quality=8,ffmpeg_params=['-threads','2']) as w:
  def still(frame,seconds,label):
   nonlocal frames
   timeline.append({'start_seconds':frames/15,'label':label,'seconds':seconds})
   for _ in range(int(seconds*15)):w.append_data(frame);frames+=1
  still(card('印刷後の回収と残留物検査',[
   '次の印刷までに残る「取り出し」と「プレート確認」を試す。',
   'SO-101＋自作ヘラ＋固定カメラ・ArUco。',
   'AIは画像検査を担当。アームは決めた経路を動く。',
   '全編シミュレーション。冷却後に接着が解けたPLAを想定。']),8,'intro')
  layout=Image.open(O/'layout.png').convert('RGB');draw=ImageDraw.Draw(layout);draw.rectangle((0,0,1280,55),fill='white');draw.text((20,12),'独自の簡略形状 / 正確なA1 CADではない',font=ImageFont.truetype(FONT,28),fill='black');still(np.array(layout),5,'workcell')
  for name in ['motion_001','motion_040','motion_000']:
   row=next(x for x in m['results'] if x['id']==name);ok=row['success'];outcome='成功：箱に回収' if ok else '失敗：回収できず'
   selected='評価後に追加した最初の成功例' if ok else ('事前に選んだ代表条件' if name=='motion_040' else '事前ルールで選んだ最初の失敗')
   still(card(name+' / '+outcome,[f'質量{row["mass_kg"]*1000:g}g / プレート・シュートの摩擦{row["friction"]:g}',f'位置ずれ：押す方向{row["along_offset_m"]*1000:+g}mm、横{row["cross_offset_m"]*1000:+g}mm',selected+'。映像は等速。']),4,name+' title')
   start=frames
   with imageio.get_reader(str(E/f'{name}.mp4')) as reader:
    for frame in reader:
     im=Image.fromarray(frame);d=ImageDraw.Draw(im);d.rectangle((0,675,1280,720),fill='white');d.text((16,681),outcome+' / 等速・固定経路 / 簡略形状・接着なし',font=ImageFont.truetype(FONT,23),fill='black');w.append_data(np.array(im));frames+=1
   timeline.append({'start_seconds':start/15,'label':name,'seconds':(frames-start)/15})
  still(card('回収の成功は15/81条件',[
   '摩擦0.1：15/27成功。摩擦0.3・0.6：各0/27。',
   '箱に届いた17条件のうち、2条件は接触で失敗とした。',
   '低摩擦でも位置ずれに弱い。固定経路だけでは回収しきれない。',
   '係数は試験用の仮定。実機での成功確率ではない。']),8,'motion results')
  still(card('小片の検出を改善するため、CNNを追加学習',[
   '3層・1,657パラメータ。基準は学習180枚＋重み選択60枚。',
   '小片・影・低露光を追加し、605パッチで学習、145で重み選択。',
   '補正画像は128→512画素。撮影元の1280×720画素は同じ。',
   '別の280枚でしきい値を選び、未使用560枚で比較した。']),10,'AI training')
  for name in ['tiny_fragments','bed_similar','unseen_hues']:
   still(np.array(Image.open(O/f'vision/example_{name}.png')),6,'example '+name)
  still(card('見逃しと不要停止を、両方数える',[
   f'残留物280枚の見逃し：基準CNN {a["false_clear"]} / 追加学習後 {b["false_clear"]} / 色判定 {r["false_clear"]}',
   f'空280枚の不要停止：基準CNN {a["false_block"]} / 追加学習後 {b["false_block"]} / 色判定 {r["false_block"]}',
   f'小片40枚の見逃しは、追加学習後も{tiny["false_clear"]}枚残った。',
   f'位置補正できない{b["rejected"]}枚も除外せず、停止として採点。',
   '解像度・しきい値・追加学習の6手法比較はPDFに掲載。']),12,'AI results')
  still(card('自動で印刷再開するには、まだ足りない',[
   '回収は摩擦と位置に依存。画像検査にも見逃しが残った。',
   '実機、接着の剥離、温度・力の計測は未検証。',
   '制限付きCADを除き、公開可能な形状・新しいデータで再評価。',
   'コード・重み・条件・全結果・再現手順を同梱。',
   '生成AIを設計・実装・草案に使用。会社資産は使用していない。']),9,'conclusion')
 assert 60<=frames/15<=180,frames/15
 (O/'video_timeline.json').write_text(json.dumps({'fps':15,'frames':frames,'seconds':frames/15,'segments':timeline},indent=2)+'\n');print('video seconds',frames/15)
if __name__=='__main__':main()
