import numpy as np, time
def period_q_orbit(K, p, q, I0, th0=0.0, maxit=80, tol=1e-12):
    x=np.array([th0,I0],float); target=np.array([2*np.pi*p,0.0])
    def Fq(x):
        th,I=x; M=np.eye(2)
        for _ in range(q):
            s,c=np.sin(th),np.cos(th)
            M=np.array([[1+K*c,1.0],[K*c,1.0]])@M
            In=I+K*s; th,I=th+In,In
        return np.array([th,I]),M
    conv=False
    for it in range(maxit):
        y,M=Fq(x); resid=y-x-target
        if np.linalg.norm(resid)<tol: conv=True; break
        try: dx=np.linalg.solve(M-np.eye(2),resid)
        except np.linalg.LinAlgError: break
        r0=np.linalg.norm(resid); step=1.0; xn=x-dx
        for _ in range(25):
            y2,_=Fq(xn); r2=np.linalg.norm(y2-xn-target)
            if r2<r0: break
            step*=0.5; xn=x-step*dx
        x=xn
    y,M=Fq(x); return x,M,conv
def R(K,p,q,n_starts=8):
    I_center=2*np.pi*p/q; best=None
    for i in range(n_starts):
        th0=2*np.pi*i/n_starts
        for dI in np.linspace(-0.3,0.3,3):
            x,M,conv=period_q_orbit(K,p,q,I_center+dI,th0)
            if not conv or abs(x[1]-I_center)>0.8: continue
            r=(1-np.trace(M)/2)**2/4
            if best is None or r<best: best=r
    return best
t0=time.time()
# table
for K in [0.90,0.95,0.9716,0.98,1.00,1.05]:
    row=f"K={K:.4f}: "
    for (p,q) in [(3,5),(5,8),(8,13),(13,21)]:
        r=R(K,p,q)
        row+=f"q={q}:{r if r is None else round(r,3)} "
    print(row,flush=True)
# bisection on plateau: R21 - R13 sign change
def diff(K):
    a=R(K,13,21); b=R(K,8,13)
    if a is None or b is None: return None
    return a-b
lo,hi=0.96,0.985
for _ in range(16):
    mid=0.5*(lo+hi); d=diff(mid); dlo=diff(lo)
    if d is None or dlo is None or dlo*d>0: lo=mid
    else: hi=mid
print(f"K_c ~ {0.5*(lo+hi):.5f}  (lit 0.97163)",flush=True)
print(f"total time {time.time()-t0:.1f}s")
