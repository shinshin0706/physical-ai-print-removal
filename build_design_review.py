"""Build the public submission from recorded results; no restricted CAD renders."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
ROOT=Path(__file__).parent; O=ROOT/'outputs/submission';SUB=ROOT/'submission'
NAMES={'baseline128':'基準CNN・128画素','threshold128':'128画素・しきい値調整','resolution512':'512画素・面積を維持','tuned512':'512画素・しきい値調整','improved512':'512画素・追加学習','rule512':'色判定・512画素'}
GROUP_NAMES={'reference':'通常','bed_similar':'プレートに近い色','achromatic':'白・灰・黒','unseen_hues':'未学習の色','tiny_fragments':'1.5〜5mmの小片','directional_shadow':'強い影','low_exposure':'低露光'}

def build():
 SUB.mkdir(exist_ok=True)
 m=json.loads((O/'motion/motion_results.json').read_text());v=json.loads((O/'vision/final_comparison.json').read_text());sel=json.loads((ROOT/'model/submission/final_selection.json').read_text());a=v['methods']['baseline128']['overall'];b=v['methods']['improved512']['overall'];tiny=v['methods']['improved512']['by_group']['tiny_fragments'];rule=v['methods']['rule512']['overall']
 pdfmetrics.registerFont(TTFont('JP','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
 c=canvas.Canvas(str(SUB/'report.pdf'),pagesize=A4);c.setTitle('印刷後の回収と残留物検査');c.setAuthor('');W,H=A4
 body=ParagraphStyle('body',fontName='JP',fontSize=10.5,leading=16);small=ParagraphStyle('small',parent=body,fontSize=8.5,leading=12.5);texts=[];page=0
 def start(title):
  nonlocal page
  if page:c.showPage()
  page+=1;c.setFont('JP',8);c.drawString(42,H-28,'Physical AI 応用編1 / シミュレーション実験 / 2026-11-01');c.setFont('JP',18);c.drawString(42,H-64,title);texts.append(title);c.setFont('JP',8);c.drawRightString(W-42,22,f'{page} / 5');return H-88
 def para(text,y,style=body):
  p=Paragraph(text,style);_,h=p.wrap(W-84,y);p.drawOn(c,42,y-h);texts.append(text);return y-h-11
 def pic(path,y,h):
  c.drawImage(str(path),42,y-h,width=W-84,height=h,preserveAspectRatio=True,anchor='c');return y-h-12
 def table(rows,y,widths):
  texts.extend([' | '.join(map(str,r)) for r in rows]);t=Table([[Paragraph(str(x),small) for x in r] for r in rows],colWidths=widths);t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f7')),('LINEBELOW',(0,0),(-1,-1),.3,colors.lightgrey),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));_,h=t.wrap(W-84,y);t.drawOn(c,42,y-h);return y-h-12
 def end(y):
  if y<40:raise RuntimeError(f'Page {page} overflow: {y}')
 y=start('印刷が終わった後の、人の手を減らしたい')
 y=para('造形物を取り出し、プレートに何も残っていないか確認する。この2つを小型アームとカメラで行う構成を作った。Bambu A1とPLAの利用を想定し、今回は冷却後に接着が解けた物だけを対象にした。強く付着した造形物を剥がす動作は含めていない。',y)
 y=pic(O/'layout.png',y,260)
 y=para(f'動作は81条件中{m["success"]}条件で成功した。画像検査では小型CNNを学習し、同じ560枚で比較した。残留物280枚の見逃しは、基準の{a["false_clear"]}枚から追加学習後の{b["false_clear"]}枚になった。空280枚での不要停止は{a["false_block"]}枚から{b["false_block"]}枚に変わった。',y)
 y=para('アームは決めた経路を動き、AIは最後の残留物検査を担当する。物体を見つけて経路を修正する制御や、学習によるアーム操作は実装していない。画像検査の改善と、回収動作の成立範囲を分けて調べた。',y)
 y=para('図のプリンターは独自に作った簡略形状で、A1の正確なCADではない。動作・画像とも合成データであり、実機では未検証。',y,small);end(y)
 y=start('外部センサーは、固定カメラ1台に絞る')
 y=table([['構成','今回の設定'],['アーム','白いSO-101。公開モデルを使用。出力上限0.981NmはST3215・12V版の仕様を参考に設定'],['工具','自作樹脂ヘラ120×40mm。根元6mm、先端0.8mm。STLとSTEPを同梱。把持は剛体として近似'],['台・回収先','1200×900mm、高さ740mmの台。幅200mmのシュートと下段の箱'],['撮影','C920クラスの固定RGBカメラを想定。1280×720画素、対角画角78度のピンホール近似'],['位置補正','ArUcoを4枚配置し、3枚以上を検出した時だけプレート面を補正。検出不足なら停止']],y,[85,426])
 y=para('画像の処理は「RGB撮影 → マーカー検出 → プレート面の切り出し → CNN → 残留画素数で判定」の順に行う。実行時の入力はRGB画像・カメラ内部パラメータ・歪み係数だけで、物体位置や正解マスクは渡さない。',y)
 y=para('128×128への縮小で小片が数画素になる問題があったため、切り出しを512×512へ変更した。元の撮影画像は同じであり、カメラ自体の解像度を増やしたわけではない。レンズ校正と実画像での評価は未実施。',y)
 y=para('SO-101を選んだのは、小型で公開モデルがあり、低抵抗で離れた部品の回収を試しやすいため。今回のシミュレーションでは追従できたが、これだけで実機のトルクや工具保持力が足りるとは判断しない。温度・接着力・把持力はカメラから測っていない。',y)
 y=para('プリンターの造形面256×256mmは公開仕様を参照した。内部形状、退避高さ、設置寸法は設計上の仮定である。実機への導入には、ベッドとヘッドの退避位置、工具固定、干渉と停止方法の確認が残る。',y);end(y)
 y=start('回収動作は、低摩擦でも取りこぼした')
 y=para('30×30×20mmの部品に対し、質量4・8・20g、摩擦0.1・0.3・0.6、押す方向のずれ−15・0・＋15mm、横ずれ−20・0・＋20mmを組み合わせた。81条件すべて同じ経路で110mm押し、その後80mm上へ退避する。位置に応じた修正はしない。',y)
 y=table([['接触摩擦','箱へ回収','成功条件を満たす'],*[[str(x['friction']),f"{x['collected']}/27",f"{x['success']}/27"] for x in m['by_factor']['friction']],['全条件',f"{m['collected']}/81",f"{m['success']}/81"]],y,[171,170,170])
 y=para('成功は、最後の0.5秒間に物体全体が箱内にあり、並進速度1cm/s未満、角速度0.1rad/s未満、関節誤差3度未満、アームと固定物の接触力が0.01N以下の時とした。採点にはシミュレーターの正解状態を使用し、制御には使用していない。',y)
 y=para(f'箱に届いたのは17条件だが、そのうち2条件は固定物への接触で失敗とした。全体で接触による失敗は6条件。関節誤差は最大{max(r["max_joint_error_deg"] for r in m["results"]):.2f}度で、追従の基準は全条件で満たした。低摩擦でも横ずれなどで回収できず、経路を固定する方式には限界がある。',y)
 y=para('摩擦は部品とプレート、部品とシュートの両方に同じ値を指定した。他の面は固定値のまま。どちらの面が結果を支配するかは、今回の実験だけでは分からない。これらの係数は実測値ではなく、現実の成功確率を表す81試行でもない。',y)
 y=para('動画には事前に決めた代表条件motion_040、最初の失敗motion_000、評価後に追加した最初の成功motion_001を収録した。以前のCADを使った結果は、形状が違うため今回の結果と改善率として比較しない。',y,small);end(y)
 y=start('小片の追加学習を、未使用の560枚で比較')
 y=para('モデルは3層・1,657パラメータのCNN。合成画像240枚を180枚の学習用と60枚の重み選択用に分け、Adam・学習率0.003・120エポックで学習した。次に通常・小片・影・低露光の320枚から切り出した605パッチで追加学習し、別の145パッチで重みを選んだ。追加学習は学習率0.001・50エポック。',y)
 y=para('判定しきい値は別の検証画像280枚で選んだ。「空画像の不要停止を10%以下」に収め、その範囲で見逃しを最小にした。512画素の基準CNNは候補内で10%以下にできず、不要停止が最少の設定を採用した。途中で得た常時停止に近い設定は検証段階で棄却した。最終モデルとしきい値を固定した後、別seedの560枚を生成した。最終評価後の調整はしていない。',y)
 y=table([['方法','見逃し / 280','不要停止 / 280','判定画素数'],*[[NAMES[k],x['overall']['false_clear'],x['overall']['false_block'],x['minimum_pixels']] for k,x in v['methods'].items()]],y,[236,92,100,83])
 y=para('128画素の基準は20画素以上で残留物ありとする。512画素の「面積を維持」は320画素以上とし、切り出しサイズだけを変えた。しきい値調整と追加学習を別行に分け、解像度変更だけの効果と区別した。色判定も同じ検証画像でしきい値を選んだ。',y)
 y=para(f'全手法で検査不可は{b["rejected"]}/560枚。これらは空と判定せず停止に含めた。物体が補正画像で消えた場合も、有無判定の分母に残した。検証時の不要停止10%という条件は、未知画像での保証ではない。',y,small);end(y)
 y=start('何が改善し、何がまだできないか')
 y=table([['条件（残留物各40枚）','基準CNN見逃し','追加学習後','色判定'],*[[GROUP_NAMES[g],v['methods']['baseline128']['by_group'][g]['false_clear'],v['methods']['improved512']['by_group'][g]['false_clear'],v['methods']['rule512']['by_group'][g]['false_clear']] for g in GROUP_NAMES]],y,[227,102,91,91])
 y=para(f'小片の見逃しは追加学習後も{tiny["false_clear"]}/40枚残った。全体での見逃しは追加学習後{b["false_clear"]}/280枚、色判定{rule["false_clear"]}/280枚だった。検査で空と出ても、次の印刷を自動で始める判断には使えない。プレートが空であることと、部品が箱に回収されたことも別の状態として扱う。',y)
 y=para('この課題では、機材と観測の設計、合成データによる学習、学習用と評価用の分離、単純手法との比較、失敗条件の分析を実装した。ROS 2やACTは使用していない。講座で扱うシステム設計とモデル評価を、印刷後の取り出しという場面に適用した検討である。',y)
 y=para('公開用には、配布制限のあるMakerWorldのA1 CAD・派生画像・旧重みを含めず、独立した簡略形状と新しい学習結果を使った。SO-101はApache-2.0の公開モデルで、ライセンスと版を同梱。自作部分はMITライセンス。会社の資産は使用していない。',y)
 y=para('設計案・実装・文章草案には生成AIを使用した。寸法の仮定、選んだ数値、データ生成seed、重み、全結果、再現コマンドはコードとREADMEに残した。実機試験、実測の摩擦・剥離力、カメラ校正は今後の確認事項である。',y)
 y=para('出典：<link href="https://bambulab.com/en/a1/tech-specs" color="#205080">Bambu A1仕様</link> ／ <link href="https://github.com/google-deepmind/mujoco_menagerie" color="#205080">MuJoCo Menagerie・SO-101</link> ／ <link href="https://docs.waveshare.com/ST3215_Servo" color="#205080">ST3215</link> ／ <link href="https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html" color="#205080">OpenCV ArUco</link>。カメラ・箱・各資産の出典と利用条件はTHIRD_PARTY_NOTICES.mdに記載。',y,small);end(y)
 c.save();(SUB/'report_text.txt').write_text('\n\n'.join(texts)+'\n');print('report: 5 pages')
if __name__=='__main__':build()
