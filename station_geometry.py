"""Parametric preliminary workstation, metres; NOT a measured A1 CAD model.
Only A1 overall W/D/H, build area and required Y workspace are manufacturer facts.
Internal component envelopes and parked positions MUST be replaced/checked against
CAD and the physical unit. Solid boxes are conservative design placeholders, not
a claim of tolerance. No external restricted CAD is redistributed.
"""
TABLE_TOP=.740
PLATE_TOP=.840
BED_CENTER=(0.,-.195)
BENCH_FRONT=-.450
BIN_CENTER=(0.,-.635)
SOURCES={
'a1_specs':'https://cdn1.bambulab.com/documentation/quick-start-a75adcb1d5d5e/Quick%20Start%20Guide%20for%20A1.pdf',
'a1_clearance':'https://wiki.bambulab.com/en/a1/manual/faq',
'candidate_cad':'https://makerworld.com/en/models/770920-bambulab-a1-cad-step-model',
'box':'https://www.ikea.com/jp/ja/p/trofast-storage-box-light-green-70582090/'}

def components(parked=True):
 # name, centre, full dimensions, RGBA, source status
 c=[]
 def box(n,p,s,col=(.68,.69,.70,1),status='design_assumption'):
  c.append(dict(name=n,center_m=p,size_m=s,rgba=col,status=status))
 box('bench_top',(-.20,0,.731),(1.2,.9,.018),(.63,.49,.33,1))
 for x in [-.75,.35]:
  for y in [-.40,.40]:box(f'bench_leg_{x}_{y}',(x,y,.36),(.04,.04,.72),(.25,.26,.28,1))
 box('bin_shelf',(0,-.625,.54),(.48,.38,.018),(.63,.49,.33,1))
 for x in [-.22,.22]:box(f'shelf_bracket_{x}',(x,-.45,.62),(.025,.025,.2),(.3,.3,.3,1))
 box('a1_base',(0,0,.777),(.34,.410,.074),(.82,.83,.84,1))
 for x in [-.2125,.2125]:box(f'a1_column_{x}',(x,.035,.955),(.04,.055,.430),(.75,.76,.77,1))
 box('a1_top',(0,.035,1.155),(.465,.055,.03),(.75,.76,.77,1))
 box('a1_x_beam',(0,.015,1.070 if parked else .920),(.405,.05,.045),(.76,.77,.78,1))
 box('a1_head',(.12,.0,1.025 if parked else .875),(.07,.10,.09),(.25,.25,.27,1))
 box('a1_display',(.215,-.14,.80),(.09,.10,.045),(.1,.1,.12,1))
 box('a1_y_rail',(0,0,.800),(.085,.41,.045),(.25,.26,.28,1))
 box('a1_bed_carriage',(0,-.195,.821),(.245,.245,.03),(.3,.3,.32,1))
 box('build_plate',(0,-.195,.838),(.256,.256,.004),(.66,.53,.26,1),'manufacturer_build_area; assumed_top_height')
 # TROFAST external dimensions are verified. Inner wall/bottom dimensions assumed.
 box('bin_floor',(0,-.635,.553),(.420,.300,.008),(.77,.80,.75,1),'manufacturer_outer_size; assumed_wall')
 for n,p,s in [('front',(0,-.782,.599),(.42,.006,.100)),('back',(0,-.488,.599),(.42,.006,.100)),('left',(-.207,-.635,.599),(.006,.288,.100)),('right',(.207,-.635,.599),(.006,.288,.100))]:box('bin_'+n,p,s,(.77,.80,.75,1),'manufacturer_outer_size; assumed_wall')
 return c
