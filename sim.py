"""End-to-end synthetic check: simulate UM-BMID-like Gen-2 S21 data with the documented
geometry, then reconstruct with the repo pipeline (as-is) and with geometry fixes."""
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from src import physics as ph, pipeline as pl
C=3e8
F=np.linspace(1e9,8e9,1001)
R_RAW_CM=21.5
R_TRUE=0.97*(R_RAW_CM/100-0.106)+0.148          # phase-centre radius, metres
ANG=np.deg2rad(-130)-np.arange(72)*np.deg2rad(5)  # Port A, clockwise
ANG_B=ANG+np.deg2rad(60)

def legs(a,b,R):
    d=b-a; A=(d*d).sum(-1); B=2*(a*d).sum(-1); Cc=(a*a).sum(-1)-R*R
    disc=B*B-4*A*Cc; L=np.sqrt(A); s=np.sqrt(np.maximum(disc,0))
    t1=np.clip((-B-s)/(2*A),0,1); t2=np.clip((-B+s)/(2*A),0,1)
    tin=np.where(disc>0,(t2-t1)*L,0.0); return L-tin,tin

def exact_delay(txx,txy,rxx,rxy,px,py,R,v):
    P=np.stack([px,py],-1)
    out=np.zeros((len(txx),len(px)))
    for i in range(len(txx)):
        tx=np.array([txx[i],txy[i]]); rx=np.array([rxx[i],rxy[i]])
        a1,t1=legs(np.broadcast_to(tx,P.shape),P,R); a2,t2=legs(P,np.broadcast_to(rx,P.shape),R)
        out[i]=(a1+a2)/C+(t1+t2)/v
    return out

def simulate(tum_xy_m, R, v, skin_amp=4.0, noise=0.02, rng=None):
    tx=np.c_[R_TRUE*np.cos(ANG),R_TRUE*np.sin(ANG)]; rx=np.c_[R_TRUE*np.cos(ANG_B),R_TRUE*np.sin(ANG_B)]
    pts=[np.array(tum_xy_m)]; amps=[1.0]
    # skin/boundary: specular-ish points on the circle facing each antenna pair -> model as ring of scatterers
    th=np.linspace(0,2*np.pi,180,endpoint=False)
    ring=np.c_[R*np.cos(th),R*np.sin(th)]
    S=np.zeros((len(F),72),complex)
    tau_t=exact_delay(tx[:,0],tx[:,1],rx[:,0],rx[:,1],np.array([tum_xy_m[0]]),np.array([tum_xy_m[1]]),R,v)[:,0]
    S+=np.exp(-2j*np.pi*F[:,None]*tau_t[None,:])
    tau_r=exact_delay(tx[:,0],tx[:,1],rx[:,0],rx[:,1],ring[:,0],ring[:,1],R*0.999,v)  # (72,180)
    # weight ring points by how much they face the bisector of the antenna pair (specular approximation)
    mid=(tx+rx)/2; mid/=np.linalg.norm(mid,axis=1,keepdims=True)
    w=np.clip(ring@mid.T/R,0,None).T**20      # (72,180)
    w/=w.sum(1,keepdims=True)
    S+=skin_amp*np.einsum('ak,fak->fa',w,np.exp(-2j*np.pi*F[:,None,None]*tau_r[None]))
    rng=rng or np.random.default_rng(0)
    S+=noise*(rng.standard_normal(S.shape)+1j*rng.standard_normal(S.shape))
    return S

def fixed_geometry(ant_rad_mm, n_ant=72, separation_deg=60.0, apply_correction=True):
    r=(0.97*(ant_rad_mm/1000-0.106)+0.148) if apply_correction else ant_rad_mm/1000
    a=np.deg2rad(-130)-np.arange(n_ant)*np.deg2rad(360/n_ant); b=a+np.deg2rad(separation_deg)
    return dict(ant_x=r*np.cos(a),ant_y=r*np.sin(a),ant_x_b=r*np.cos(b),ant_y_b=r*np.sin(b),ant_rad_m=r)

def fixed_snell(ax_,ay,bx,by,gx,gy,R,v,v_air=C): return exact_delay(ax_,ay,bx,by,gx,gy,R,v)

ORIG=(ph.get_antenna_geometry, ph.snellius_bistatic_delay_precise)
def set_mode(mode):
    pl._DELAY_CACHE.clear()
    g,s=ORIG
    if mode in ('fix_all','fix_radius_angle'): g=fixed_geometry
    if mode=='fix_all': s=fixed_snell
    ph.get_antenna_geometry=g; ph.snellius_bistatic_delay_precise=s

def run(mode, cases, R_mm=45.0, fat=0.95, fib=0.05):
    set_mode(mode)
    eps=fat*7+fib*45; v=C/np.sqrt(eps)
    rows=[]
    for k,(x,y) in enumerate(cases):
        s21=simulate((x/1000,y/1000),R_mm/1000,v,rng=np.random.default_rng(k))[None]
        tm=pd.DataFrame([dict(breast_radius_mm=R_mm,fat_fraction=fat,fib_fraction=fib,ant_rad=R_RAW_CM,
              original_s21_idx=0,emp_ref_id=np.nan,phant_id=f'SIM{k}',tumor_x_mm=x,tumor_y_mm=y,tumor_radius_mm=7.5)])
        r=pl.reconstruct_scan(0,s21,tm,{},freq_axis=F,use_tvsvd=True,use_rms_norm=True,bandpass_mode='full',gate_ns=0.70)
        rows.append(dict(mode=mode,x=x,y=y,gt_r=np.hypot(x,y),pred_r=np.hypot(r['peak_x_mm'],r['peak_y_mm']),LE=r['localization_error_mm']))
    return pd.DataFrame(rows)

if __name__=='__main__':
    rng=np.random.default_rng(1)
    rr=rng.uniform(5,38,16); th=rng.uniform(0,2*np.pi,16)
    cases=list(zip(rr*np.cos(th),rr*np.sin(th)))
    res=pd.concat([run(m,cases) for m in ('as_is','fix_radius_angle','fix_all')])
    res.to_csv('sim_results.csv',index=False)
    print(res.groupby('mode')[['LE','pred_r','gt_r']].agg(['mean','median']).round(1))
    print(res.groupby('mode').apply(lambda d:(d.LE<=15).mean()).rename('det@15'))
