import sys,json,csv,math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

OUT=Path(sys.argv[1]);FPS=25
POLICIES=['P0','P1','P2']
COLORS={'P0':(39,110,205),'P1':(231,119,32),'P2':(211,50,64)}
LEFT=(31,157,182);RIGHT=(214,135,73)
W,H=1680,832
PANEL_W=W//3

try:
    base='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    f12=ImageFont.truetype(base,15)
    f15=ImageFont.truetype(base,18)
    f24=ImageFont.truetype(bold,24)
    f32=ImageFont.truetype(bold,31)
except OSError:
    f12=ImageFont.load_default();f15=f12;f24=f12;f32=f12

D={};meta={}
for p in POLICIES:
    D[p]=np.load(OUT/('g1_pose_'+p+'_1p4_rep0.npz'))
    meta[p]=json.loads((OUT/('g1_pose_info_'+p+'.json')).read_text(encoding='utf-8'))
    assert len(D[p]['times']) > 20, p
    assert D[p]['body_xyz'].shape[0]==D[p]['root_xyz'].shape[0]
    assert meta[p]['command_mps']==1.4 and meta[p]['rep']==0
    assert meta[p]['fps']==FPS
frames=min(len(D[p]['times']) for p in POLICIES)
print('Render frames:',frames, 'seconds',float(min(D[p]['times'][-1] for p in POLICIES)))

# Use the SAME fixed map axes for all three policies. Local coordinates start at (0,0).
paths={p:D[p]['root_xyz'][:,:2]-D[p]['start_root_xy'][None,:] for p in POLICIES}
xall=np.hstack([paths[p][:,0] for p in POLICIES]+[np.array([0.])])
yall=np.hstack([paths[p][:,1] for p in POLICIES]+[np.array([0.])])
xmin=float(min(-1,xall.min()-1));xmax=float(max(28,xall.max()+1.0))
ymin=float(min(-5,yall.min()-1));ymax=float(max(5,yall.max()+1))
scale=min(472/(xmax-xmin),277/(ymax-ymin))
# Center plot to avoid different aspect distortion.
xmid=(xmin+xmax)/2;ymid=(ymin+ymax)/2
left_labels=[]

# Draw simplified bone chains from measured rigid-body LINK centers.
for p in POLICIES:
    names=meta[p]['body_names'];idx={n:i for i,n in enumerate(names)}
    chains=[]
    for side,col in [('left',LEFT),('right',RIGHT)]:
        leg=[x for x in ['pelvis',f'{side}_hip_yaw_link',f'{side}_hip_roll_link',f'{side}_hip_pitch_link',
                         f'{side}_knee_link',f'{side}_ankle_pitch_link',f'{side}_ankle_roll_link'] if x in idx]
        arm=[x for x in ['torso_link',f'{side}_shoulder_pitch_link',f'{side}_shoulder_roll_link',
                         f'{side}_shoulder_yaw_link',f'{side}_elbow_link',f'{side}_wrist_roll_link'] if x in idx]
        if len(leg)>=2:chains.append((leg,col))
        if len(arm)>=2:chains.append((arm,col))
    torso=[x for x in ['pelvis','waist_yaw_link','waist_roll_link','torso_link'] if x in idx]
    if len(torso)>=2:chains.append((torso,(76,86,106)))
    meta[p]['chains']=[([idx[n] for n in nodes],c) for nodes,c in chains]


def text(d,xy,s,fill=(41,52,70),font=f15):
    d.text(xy,s,fill=fill,font=font)

def poly_arrow(draw,x,y,angle,color,r=19):
    # Positive world yaw points positive Y (up on diagram).
    ux,uy=math.cos(angle),-math.sin(angle)
    x2=x+r*ux;y2=y+r*uy
    draw.line((x,y,x2,y2),fill=color,width=5)
    k=8
    nx,ny=-uy,ux
    draw.polygon([(x2+7*ux,y2+7*uy),(x2-8*ux+5*nx,y2-8*uy+5*ny),
                  (x2-8*ux-5*nx,y2-8*uy-5*ny)],fill=color)

def draw_at(i):
    t=float(D[POLICIES[0]]['times'][i])
    im=Image.new('RGB',(W,H),(245,247,250));dr=ImageDraw.Draw(im)
    dr.rectangle((0,0,W,76),fill=(24,36,54))
    text(dr,(27,15),'UNITREE G1  /  1.4 m/s  /  Isaac Gym',fill=(255,255,255),font=f32)
    text(dr,(W-290,24),f't = {t:5.2f} s   /   18 s',fill=(210,223,237),font=f15)
    for col,p in enumerate(POLICIES):
        px=col*PANEL_W
        accent=COLORS[p]
        dr.rounded_rectangle((px+8,82,px+PANEL_W-8,H-11),radius=14,fill=(255,255,255),outline=(219,227,236),width=2)
        dr.rectangle((px+17,90,px+24,124),fill=accent)
        text(dr,(px+34,91),f'{p}  |  contact tolerance '+{'P0':'0 ms','P1':'20 ms','P2':'40 ms'}[p],font=f24)
        # Fixed-world XY path plot
        box=(px+30,165,px+PANEL_W-30,450)
        dr.rounded_rectangle(box,radius=9,fill=(249,251,253),outline=(228,232,238),width=2)
        ox=px+PANEL_W/2;oy=304
        def tr(x,y):return (int(ox+scale*(x-xmid)),int(oy-scale*(y-ymid)))
        for gx in range(math.floor(xmin/5)*5,math.ceil(xmax/5)*5+1,5):
            a=tr(gx,ymin);b=tr(gx,ymax)
            dr.line((a,b),fill=(231,236,241),width=1)
        for gy in range(math.floor(ymin/5)*5,math.ceil(ymax/5)*5+1,5):
            a=tr(xmin,gy);b=tr(xmax,gy)
            dr.line((a,b),fill=(231,236,241),width=1)
        # Desired straight line y=0, initial command +X
        a=tr(xmin,0);b=tr(xmax,0)
        # dashed desired path
        for s in range(a[0],b[0],18):dr.line((s,a[1],min(s+10,b[0]),a[1]),fill=(164,173,186),width=2)
        curve=paths[p][:i+1]
        if len(curve)>1:dr.line([tr(float(x),float(y)) for x,y in curve],fill=accent,width=5,joint='curve')
        last=curve[-1];cx,cy=tr(last[0],last[1])
        dr.ellipse((cx-8,cy-8,cx+8,cy+8),fill=accent,outline=(255,255,255),width=2)
        yaw=float(D[p]['yaw_rad'][i])
        poly_arrow(dr,cx,cy,yaw,accent)
        xval=float(last[0]);yval=float(last[1]);deg=math.degrees(yaw)
        text(dr,(px+35,138),'TOP VIEW  |  World coordinates  |  grid 5 m',font=f15)
        text(dr,(px+36,463),f'X = {xval:5.1f} m    Y = {yval:+5.1f} m    yaw = {deg:+6.1f} deg',font=f15)
        # enlarged local X-Z skeleton showing gait motion, rotated into current heading
        sx0=px+28;sy0=517;sx1=px+PANEL_W-28;sy1=764
        dr.rounded_rectangle((sx0,sy0,sx1,sy1),radius=10,fill=(252,253,254),outline=(231,236,242),width=2)
        text(dr,(px+40,497),'SIDE VIEW  |  Robot heading frame',font=f15)
        ground=sy1-24
        dr.line((sx0+12,ground,sx1-12,ground),fill=(166,173,184),width=2)
        pos=D[p]['body_xyz'][i];root=D[p]['root_xyz'][i]
        cos=math.cos(yaw);sin=math.sin(yaw)
        def side_proj(pt):
            dx=float(pt[0]-root[0]);dy=float(pt[1]-root[1]); fwd=cos*dx+sin*dy
            return (int(px+PANEL_W/2+175*fwd),int(ground-175*float(pt[2])))
        for js,color in meta[p]['chains']:
            pts=[side_proj(pos[k]) for k in js]
            dr.line(pts,fill=color,width=7,joint='curve')
            for cx2,cy2 in pts:dr.ellipse((cx2-4,cy2-4,cx2+4,cy2+4),fill=color)
        if not meta[p]['chains']:
            for pt in pos:
                a,b=side_proj(pt);dr.ellipse((a-3,b-3,a+3,b+3),fill=accent)
        text(dr,(px+34,775),'Left leg: cyan   /   Right leg: ochre',font=f12)
    return im

output=OUT/'P012_1p4mps_straightness_3panel.mp4'
preview=OUT/'P012_1p4mps_preview.png'
with imageio.get_writer(str(output),fps=FPS,codec='libx264',quality=7,macro_block_size=16) as writer:
    for i in range(frames):
        frame=draw_at(i)
        if i==min(frames-1,frames//2):frame.save(preview)
        writer.append_data(np.asarray(frame))
        if i%75==0:print('Rendered',i+1,'/',frames,flush=True)

# One row per policy; these are a SINGLE representative trial, not 8-rep medians.
rows=[]
for p in POLICIES:
    last=paths[p][frames-1]
    rows.append(dict(model=p,speed_command_mps=1.4,rep=0,
        recorded_s=float(D[p]['times'][frames-1]),
        delta_world_x_m=float(last[0]),delta_world_y_m=float(last[1]),
        final_yaw_deg=float(math.degrees(float(D[p]['yaw_rad'][frames-1]))),
        frames=frames))
with (OUT/'P012_1p4mps_trial0_video_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
print('VIDEO_DONE:',output,'size MiB',round(output.stat().st_size/1024**2,1))
print('PREVIEW:',preview)
