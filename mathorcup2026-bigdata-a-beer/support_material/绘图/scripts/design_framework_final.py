"""Four aligned task lanes with separate prediction and design dependencies."""
from repair_common import *
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse, Circle, Polygon, Wedge
from matplotlib.colors import to_rgb


def tint(c,a=.13):
 r=np.array(to_rgb(c));return tuple(1-a+a*r)

def main():
 f=figure(1);f.set_size_inches(6.7,4.9)
 ax=f.add_axes([0,0,1,1]);ax.set(xlim=(0,670),ylim=(0,490));ax.axis('off')
 def txt(x,y,s,fs=6.0,c=INK,ha='center',va='center',rot=0,weight='normal'):
  return ax.text(x,y,s,fontsize=fs,color=c,ha=ha,va=va,rotation=rot,fontweight=weight,linespacing=1.4)
 def line(a,b,c=GRAY,lw=.5,ls='-'):
  ax.plot([a[0],b[0]],[a[1],b[1]],color=c,lw=lw,ls=ls,solid_capstyle='round')
 def arrow(a,b,c=GRAY,lw=.65,ms=5):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=ms,lw=lw,color=c,shrinkA=0,shrinkB=0))
 def matrix(x,y,w=45,h=28,rows=4,cols=6,c=TEAL,mode='varied'):
  for i in range(rows):
   for j in range(cols):
    a=.13+.64*((3*i+5*j)%11)/10
    if mode=='selected':a=.67 if j in(1,3) else.10
    ax.add_patch(Rectangle((x+j*w/cols,y+i*h/rows),w/cols-.65,h/rows-.65,fc=tint(c,a),ec='none'))
 def ring(x,y,r=21):
  for k,c in enumerate([BLUE,TEAL,PURPLE]):
   for j in range(15):ax.add_patch(Wedge((x,y),r-k*5,90+j*24,90+(j+1)*24-2,width=3.7,fc=tint(c,.15+.65*((j*7+k*3)%13)/12),ec='none'))
 def graph(x,y,c=PURPLE):
  pts=[]
  for j in range(21):
   th=2*np.pi*j/21;pts.append([x+22*np.cos(th),y+22*np.sin(th)])
  for i,j in [(0,2),(0,4),(1,5),(2,6),(4,6),(7,9),(8,12),(9,11),(10,13),(14,17),(15,19),(16,20),(17,20),(1,12),(6,15),(3,17),(8,20)]:
   ax.add_patch(FancyArrowPatch(pts[i],pts[j],connectionstyle='arc3,rad=.17',arrowstyle='-',lw=.3,color=tint(c,.35)))
  for j,p in enumerate(pts):ax.add_patch(Circle(p,1.7,fc=[TEAL,PURPLE,BLUE][j//7],ec='white',lw=.2))
 def tree(x,y,c=PURPLE):
  for a,b in [((x,y+15),(x-15,y+2)),((x,y+15),(x+15,y+2)),((x-15,y+2),(x-24,y-10)),((x-15,y+2),(x-7,y-10)),((x+15,y+2),(x+7,y-10)),((x+15,y+2),(x+24,y-10))]:line(a,b,c,.6)
  for i,xx in enumerate([-24,-7,7,24]):
   for j in range(3):ax.add_patch(Rectangle((x+xx-3+j*2,y-15),1.45,3+j%2*2,fc=tint(c,.2+.25*j),ec='none'))
 def barcode(x,y,c=TEAL):
  for i in range(7):
   for j in range(16):
    if (i*11+j*7)%17>5:ax.add_patch(Rectangle((x+j*3,y+i*3),2.8,2.5,fc=c,ec='none'))
 # Strict matrix: row lanes share every column edge; no closed card outlines.
 for j,(lo,hi,c) in enumerate([(348,445,TEAL),(241,338,PURPLE),(134,231,BLUE),(27,124,RED)]):
  ax.add_patch(Rectangle((12,lo-7),600,hi-lo+7,fc='#F6F8FA',ec='none'))
  ax.add_patch(Rectangle((12,lo-7),2.3,hi-lo+7,fc=c,ec='none'))
  line((155,lo+9),(155,hi-9),'#DCE4E9',.4);line((500,lo+9),(500,hi-9),'#DCE4E9',.4)
 txt(81,473,'双源数据与尺度',7.0)
 txt(328,473,'核心数学机制',7.0)
 txt(555,473,'任务输出与验证',7.0)
 line((12,459),(612,459),'#BDCAD2',.5)
 # Lane 1: standardized and raw-score paths are intentionally parallel.
 y=392
 txt(26,430,'01',7.5,TEAL,ha='left');txt(48,430,'双轨感官预测',7.0,ha='left')
 txt(81,410,'第3章',5.6,SECONDARY)
 matrix(27,365,32,24,c=BLUE);matrix(69,365,27,24,c=TEAL);matrix(107,365,26,24,c=PURPLE)
 txt(43,355,'X',5.8,BLUE);txt(82,355,'Yz · 50维',5.8,TEAL);txt(120,355,'R · 七档',5.8,PURPLE)
 txt(80,344,'250款 / 2197次品评',5.6,SECONDARY)
 txt(210,428,'双目标表示',6.2)
 matrix(177,381,26,27,rows=4,cols=3,c=BLUE);matrix(216,381,32,27,rows=4,cols=5,c=TEAL)
 txt(190,371,'Y16',5.6,BLUE);txt(232,371,'Y50',5.6,TEAL)
 arrow((252,395),(271,395))
 txt(321,428,'训练内分组凸融合',6.2)
 for i,v in enumerate([.24,.46,.63,.83]):
  yy=380+i*7;line((284,yy),(356,yy),'#D0DBE2',.4);ax.add_patch(Circle((284+72*v,yy),1.85,fc=TEAL,ec='white',lw=.25))
 txt(321,364,'误差互补分解',5.7,SECONDARY)
 line((361,404),(369,415),TEAL,.5);arrow((369,415),(515,415),TEAL,.5)
 # Third operator has its own raw-score input, separate from the fusion path.
 line((374,356),(374,419),'#DCE4E9',.4)
 txt(437,428,'独立原分支路',6.2,PURPLE);tree(437,393)
 txt(437,364,'R → CDF-ET',5.9,PURPLE)
 arrow((483,393),(513,393),GRAY)
 ring(539,394,21)
 for i,h in enumerate([5,11,20,25,16,9,4]):ax.add_patch(Rectangle((566+i*4,378),2.8,h,fc=tint(PURPLE,.3+.08*i),ec='none'))
 txt(553,360,'50维连续值 / 七档概率',5.9)
 txt(553,346,'宏R² · RMSE · RPS',5.6,SECONDARY)
 # Lane 2: direct observation evidence, no f* call.
 y=285
 txt(26,323,'02',7.5,PURPLE,ha='left');txt(48,323,'模块与风格内归因',7.0,ha='left')
 txt(81,303,'第4章',5.6,SECONDARY)
 matrix(30,260,37,27,c=BLUE);matrix(77,260,37,27,c=TEAL)
 txt(49,250,'X',5.8,BLUE);txt(96,250,'Yz',5.8,TEAL)
 txt(81,239,'22类风格 s',5.7,SECONDARY)
 txt(210,321,'共变模块',6.2);graph(210,282)
 txt(210,250,'99节点主连通',5.7,SECONDARY)
 arrow((249,285),(273,285))
 txt(321,321,'协方差分解',6.2)
 for xx,c,h in [(288,BLUE,26),(307,TEAL,18),(333,PURPLE,13),(352,GRAY,20)]:
  ax.add_patch(Rectangle((xx-5,267),9,h,fc=tint(c,.7),ec='none'))
 txt(321,250,'去风格均值岭回归',5.7,SECONDARY)
 arrow((366,285),(389,285))
 txt(437,321,'配对分层重采样',6.2)
 for i,(xx,yy)in enumerate([(416,269),(429,275),(442,281)]):
  ax.add_patch(Rectangle((xx,yy),23,18,fc='white',ec=tint(PURPLE,.7),lw=.45));matrix(xx+3,yy+3,17,12,3,4,c=PURPLE)
 txt(437,250,'400次 / 固定 α=3',5.7,SECONDARY)
 arrow((483,285),(513,285))
 line((519,268),(590,268),'#91A1AC',.45);line((519,268),(519,307),'#91A1AC',.45)
 line((523,273),(585,303),PURPLE,.9);line((523,280),(585,285),TEAL,.9)
 for i in range(7):
  xx=525+i*9;ax.add_patch(Circle((xx,274+i*4),1.3,fc=tint(PURPLE,.5),ec='none'));ax.add_patch(Circle((xx,281+1.2*np.sin(i)),1.3,fc=tint(TEAL,.65),ec='none'))
 txt(553,253,'50属性 Top 3 候选',5.9)
 txt(553,239,'真实标签归因 · 验证优先级',5.6,SECONDARY)
 # Lane 3: interval marginal representation, shrinkage and conditional Gaussian.
 y=178
 txt(26,216,'03a',7.5,BLUE,ha='left');txt(58,216,'质量条件生成',7.0,ha='left')
 txt(81,196,'第5.1–5.3节',5.6,SECONDARY)
 matrix(31,157,43,26,c=BLUE,cols=7);txt(101,178,'225维',6.2,BLUE);txt(101,164,'直接字段',5.7,SECONDARY)
 txt(81,145,'4VG: 168/250重复下限',5.6,SECONDARY)
 txt(210,214,'经验跳跃区间',6.2)
 line((182,160),(240,160),'#91A1AC',.45);line((182,160),(182,196),'#91A1AC',.45)
 xx=[184,191,191,206,206,220,220,237];yy=[163,163,175,175,184,184,194,194];ax.plot(xx,yy,c=BLUE,lw=.9)
 line((191,156),(206,156),BLUE,1.4);txt(210,143,'潜变量区间 (a, b]',5.7,SECONDARY)
 arrow((249,178),(273,178))
 txt(321,214,'跨风格收缩',6.2)
 matrix(281,161,27,31,5,5,c=GRAY);matrix(334,161,27,31,5,5,c=BLUE)
 arrow((312,178),(329,178),BLUE,.65)
 txt(321,143,'池化 Ledoit–Wolf',5.7,SECONDARY)
 arrow((366,178),(389,178))
 txt(437,214,'截断与条件补全',6.2)
 gx=np.linspace(-1.8,1.8,65);gy=np.linspace(-1.5,1.5,65);X,Y=np.meshgrid(gx,gy);Z=np.exp(-(X*X-1.05*X*Y+Y*Y)/1.5)
 ax.contour(437+X*13,179+Y*12,Z,levels=[.15,.35,.65],colors=[BLUE],linewidths=.45)
 ax.add_patch(Rectangle((414,161),24,18,fc=tint(TEAL,.2),ec='none'));line((438,160),(438,198),RED,.6,'--');line((413,179),(461,179),RED,.6,'--')
 txt(437,143,'低维采样 + Schur补',5.7,SECONDARY)
 arrow((483,178),(513,178))
 rng=np.random.default_rng(82);pts=rng.multivariate_normal([0,0],[[1,.33],[.33,1]],65);pts=pts[(pts[:,0]<.85)&(pts[:,1]<.8)]
 ax.add_patch(Rectangle((520,158),48,36,fc=tint(TEAL,.08),ec='none'));ax.scatter(543+pts[:,0]*8,176+pts[:,1]*6,s=2.0,c=TEAL,alpha=.67,lw=0)
 line((550,155),(550,199),RED,.6,'--');line((517,181),(584,181),RED,.6,'--')
 txt(553,145,'2200份条件谱 · 50维画像',5.9)
 txt(553,131,'DMS ≤19 / 双乙酰 ≤0.09',5.6,SECONDARY)
 # Lane 4: fixed-support certificate, joint geometry, original-unit decoding.
 txt(26,109,'03b',7.5,RED,ha='left');txt(58,109,'受控稀疏迁移',7.0,ha='left')
 txt(81,89,'第5.4–5.6节',5.6,SECONDARY)
 for i in range(3):
  xx=np.linspace(0,1,21);yy=.55+.18*np.sin(xx*(5+i)+i)+.1*i;ax.plot(30+xx*52,42+yy*19,c=[BLUE,TEAL,PURPLE][i],lw=.55)
 txt(112,66,'11款源谱',6.0);txt(112,52,'→ Tripel',5.7,SECONDARY)
 txt(81,36,'编辑预算 B ∈ {60, 100}',5.6,SECONDARY)
 txt(210,107,'固定支持双证书',6.2)
 line((208,52),(208,89),RED,.55,'--')
 for i,w in enumerate([14,20,24,17]):
  yy=53+i*8;line((210,yy),(210+w,yy),BLUE,2.7);line((206-w*.72,yy+2),(206,yy+2),TEAL,2.2)
 txt(210,36,'距离下界 / 余量上界',5.7,SECONDARY)
 arrow((249,71),(273,71))
 txt(321,107,'联合几何可行域',6.2)
 ax.add_patch(Ellipse((319,70),66,34,angle=18,fc=tint(BLUE,.07),ec=BLUE,lw=.6))
 ax.add_patch(Polygon([[321,53],[340,58],[348,72],[336,85]],fc=tint(TEAL,.25),ec=TEAL,lw=.4,hatch='////'))
 line((316,89),(341,50),RED,.7)
 ax.add_patch(Circle((295,64),3,fc='white',ec=INK,lw=.6));ax.add_patch(Circle((295,64),.9,fc=INK,ec='none'));arrow((299,65),(336,72),TEAL,.7)
 txt(321,36,'距离改善 ∩ 分类余量 η',5.7,SECONDARY)
 arrow((366,71),(389,71))
 txt(437,107,'原尺度解码与复核',6.2)
 for j,v in enumerate([.4,.7,.45,.9,.25]):
  xx=410+j*11;line((xx,54),(xx,81),'#CFD9E0',.65);ax.add_patch(Circle((xx,54+27*v),2.4,fc=BLUE,ec='white',lw=.3))
 txt(437,36,'化学界限 / 陈旧味非增',5.7,SECONDARY)
 arrow((483,71),(513,71))
 barcode(514,54)
 txt(586,69,'90.1%',7.5,TEAL)
 txt(553,39,'逐字段改动 · 100次重拟合',5.9)
 txt(553,25,'B=100, η=2 联合通过率',5.6,SECONDARY)
 # Reuse bus. Its origin is explicitly the unchanged six-model baseline;
 # it neither calls the attribution lane nor substitutes for the fused model.
 ax.add_patch(Rectangle((622,47),36,390,fc='#EAF3F3',ec='none'))
 txt(640,405,'f*',9.0,BLUE)
 txt(640,378,'六模型',5.8,BLUE);txt(640,364,'基线',5.8,BLUE)
 line((640,344),(640,57),TEAL,.85)
 txt(650,249,'固定感官映射 · 生成与迁移复用',5.8,TEAL,rot=90)
 for yy in [179,72]:
  ax.add_patch(Circle((640,yy),2.3,fc=TEAL,ec='white',lw=.45));arrow((637,yy),(607,yy),TEAL,.75)
 txt(624,196,'画像',5.5,TEAL,ha='left');txt(624,89,'验收',5.5,TEAL,ha='left')
 # Units kept outside miniature axes, where they remain legible.
 txt(15,8,'结构示意；DMS单位 μg/L，双乙酰单位 mg/L；归因使用真实观测 Yz。',5.5,SECONDARY,ha='left')
 caption='四任务建模结构。01在标准化Yz上作Y16/Y50训练内分组凸融合，在2197次公开原分R上以独立CDF-ET输出七档分布；二者为平行分支。02以真实化学与感官观测识别共变模块，并以400次配对风格分层重采样比较总体和风格内Top 3候选。03a以经验区间、跨风格收缩及Schur补生成质量条件谱。03b以固定支持双证书、联合几何和原尺度复核完成稀疏迁移。生成画像与迁移感官验收复用固定六模型基线f*，不调用融合预测器。图标均为机制示意；168/250为4VG字段重复下限观测占比，90.1%为B=100、η=2策略的来源等权联合通过率。'
 sources=[DATA/'fig03_per_attribute_branches_fusion.csv',DATA/'06_network_positions.csv',DATA/'fig08_empirical_4VG_atoms.csv',DATA/'fig12_frozen_means_recomputed.csv']
 save(f,1,caption,sources,{'illustration_only':True,'task_lanes':['01','02','03a','03b'],'frozen_predictor_reuse':True,'raw_model_separate_from_z_fusion':True,'fused_model_distinct_from_f_star':True,'attribution_uses_observed_labels':True,'repeated_atom_field':'4VG','repeated_atom_n':168,'repeated_atom_denominator':250})
 print('Four-lane framework exported at 6.7 × 4.9 in.')

if __name__=='__main__':main()
