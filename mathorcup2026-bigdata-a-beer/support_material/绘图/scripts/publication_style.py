from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
BASE=Path(__file__).resolve().parent.parent
FONT=BASE/'fonts/NotoSansCJKsc.ttf'
for p in (BASE/'fonts').glob('*'):
    if p.suffix.lower() in ('.ttf','.otf'):fm.fontManager.addfont(str(p))
FAMILY=fm.FontProperties(fname=str(FONT)).get_name()
LATIN='DejaVu Sans'
INK='#263645';SECONDARY='#697783';BLUE='#526F91';TEAL='#3D9690';PURPLE='#9B8DB2';RED='#C97772';GOLD='#D6BF83';ORANGE=RED;GRAY='#ABB7C1';GRID='#E7ECEF'
GROUP_COLORS=[TEAL,PURPLE,RED]
def setup():
    plt.rcParams.update({'font.family':[LATIN,FAMILY],'font.size':6.8,'mathtext.fontset':'stixsans','axes.titlesize':7.2,'axes.labelsize':6.5,'legend.fontsize':6.0,'xtick.labelsize':6.1,'ytick.labelsize':6.1,'axes.unicode_minus':False,'axes.linewidth':.45,'axes.edgecolor':INK,'axes.labelcolor':INK,'text.color':INK,'xtick.color':SECONDARY,'ytick.color':SECONDARY,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'grid.color':GRID,'grid.linewidth':.3,'grid.linestyle':'solid','savefig.facecolor':'white','figure.facecolor':'white','svg.fonttype':'path','pdf.fonttype':42,'svg.hashsalt':'MCB2600347_composition_v4'})
def clean(ax,axis=None):
    ax.tick_params(direction='out',length=2,width=.4,pad=2)
    ax.spines[['top','right']].set_visible(False)
    for k in ['left','bottom']:ax.spines[k].set_linewidth(.45);ax.spines[k].set_color('#85919B')
    if axis:ax.grid(axis=axis,color=GRID,lw=.3)
def panel(ax,letter,title):
    ax.set_title(f'{str(letter).lower()}  {title}',loc='left',pad=6,fontsize=7.2,fontweight='normal')
def zebra(ax,n,color='#F5F7F8'):
    for i in range(n):
        if i%2==0:ax.axhspan(i-.5,i+.5,color=color,zorder=-5,lw=0)
def badge(ax,text,xy=(.97,.04),color=None,**kwargs):
    return ax.text(*xy,text,transform=ax.transAxes,ha='right',va='bottom',fontsize=6.5,color=color or SECONDARY,**kwargs)
