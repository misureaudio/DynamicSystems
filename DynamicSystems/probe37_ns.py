import numpy as np, time, scipy.sparse as sp
def make_ns2d(N, nu, A):
    ks=[(i,j) for i in range(-N,N+1) for j in range(-N,N+1) if 0<i*i+j*j<=N*N]
    n=len(ks); karr=np.array(ks,float); k2=np.hypot(karr[:,0],karr[:,1])**2
    idx={k:m for m,k in enumerate(ks)}
    rows,cols,ls,cs=[],[],[],[]
    for i,ki in enumerate(ks):
        for j,kj in enumerate(ks):
            diff=(int(ki[0]-kj[0]),int(ki[1]-kj[1]))
            if diff in idx:
                cr=float(ki[0]*kj[1]-ki[1]*kj[0])
                if cr!=0.0:
                    rows.append(i);cols.append(j);ls.append(idx[diff]);cs.append(cr/k2[j])
    rows=np.array(rows);cols=np.array(cols);ls=np.array(ls);cs=np.array(cs,float)
    fhat=np.zeros(n,complex)
    for k in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        if k in idx: fhat[idx[k]]=A/4
    dnu=nu*k2
    def M_as(om): return sp.coo_matrix((cs*om[ls],(rows,cols)),shape=(n,n)).tocsr()
    def rhs(om): return M_as(om)@om - dnu*om + fhat
    return rhs,n,ks,idx
def even_init(ks,idx,scale,seed):
    rng=np.random.default_rng(seed); om=np.zeros(len(ks),complex)
    for m,k in enumerate(ks):
        km=(-k[0],-k[1])
        if km in idx and m>idx[km]: continue
        om[m]=scale*(rng.random()+1j*rng.random())
    for m,k in enumerate(ks):
        km=(-k[0],-k[1])
        if km in idx and m<idx[km]: om[idx[km]]=np.conj(om[m])
    return om
def rk4(f,x,dt):
    k1=f(x);k2=f(x+0.5*dt*k1);k3=f(x+0.5*dt*k2);k4=f(x+dt*k3)
    return x+(dt/6.0)*(k1+2*k2+2*k3+k4)
def active_set(e,frac=0.99):
    tot=e.sum(); s=np.sort(e)[::-1]; c=np.cumsum(s)/tot
    return np.argsort(e)[::-1][:int(np.searchsorted(c,frac))+1]
def corr_dim(X, rfac=1.0, nr=16):
    """Grassberger-Procaccia correlation dimension. X: (M,D)."""
    M=D=X.shape[0]
    # sub-sample for O(M^2) cost
    if M>2000:
        idx=np.linspace(0,M-1,2000).astype(int); X=X[idx]; M=2000
    mn=X.min(0); mx=X.max(0); Xs=(X-mn)/(mx-mn)   # [0,1]^D
    # correlation integral C(r) = (1/M^2) sum_{i!=j} 1[|Xi-Xj|<r]
    # use the max-norm distance for speed
    rs=np.logspace(-3,np.log10(rfac),nr)
    Cs=[]
    # pairwise via chunking
    from scipy.spatial.distance import cdist
    for r in rs:
        # count pairs with max-norm < r
        C=0.0
        for a in range(0,M,500):
            b=Xs[a:a+500]
            d=np.max(np.abs(b[:,None,:]-Xs[None,:,:]),axis=2)
            C+=np.sum(d<r)-len(b)  # exclude i==j
        Cs.append(C/(M*(M-1)))
    Cs=np.array(Cs); rs=np.array(rs)
    # fit log C vs log r in the linear (scaling) regime: use middle points
    m=(Cs>1e-3)&(Cs<0.5)
    if m.sum()<3: return np.nan
    D2=np.polyfit(np.log(rs[m]),np.log(Cs[m]),1)[0]
    return D2
for (nu,A) in [(0.05,1.0),(0.08,1.3),(0.1,1.3),(0.15,1.5),(0.2,2.0)]:
    t0=time.time()
    rhs,n,ks,idx=make_ns2d(5,nu,A)
    om=even_init(ks,idx,0.01,1)
    for s in range(int(400/0.02)): om=rk4(rhs,om,0.02)
    eacc=np.zeros(n)
    for s in range(1500):
        om=rk4(rhs,om,0.01); eacc+=np.abs(om)**2
    aset=active_set(eacc/1500)
    traj=[]
    for s in range(4000):
        om=rk4(rhs,om,0.01)
        if s%8==0:
            sub=om[aset]; traj.append(np.concatenate([sub.real,sub.imag]))
    X=np.array(traj)
    D2=corr_dim(X)
    print(f"nu={nu} A={A}: t={time.time()-t0:.0f}s n={n} active={len(aset)} corr_dim_D2={D2:.2f} E={0.5*np.sum(np.abs(om)**2):.2f}",flush=True)
