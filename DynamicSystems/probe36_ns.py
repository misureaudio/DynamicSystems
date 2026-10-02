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
def box_dim(X, bins=[4,6,8,12,16,24]):
    mn=X.min(0); mx=X.max(0); r=(mx-mn); r[r==0]=1
    Ds=[]
    for b in bins:
        cell=np.floor((X-mn)/r*b).astype(int)
        Ds.append(np.log(np.unique(cell,axis=0).shape[0]))
    return np.polyfit(np.log(1/np.array(bins)),np.array(Ds),1)[0]
for (nu,A) in [(0.02,5.0),(0.02,8.0),(0.03,5.0),(0.05,8.0),(0.05,10.0)]:
    t0=time.time()
    rhs,n,ks,idx=make_ns2d(5,nu,A)
    om=even_init(ks,idx,0.01,1)
    for s in range(int(300/0.01)): om=rk4(rhs,om,0.01)
    eacc=np.zeros(n)
    for s in range(1500):
        om=rk4(rhs,om,0.01); eacc+=np.abs(om)**2
    aset=active_set(eacc/1500)
    traj=[]
    for s in range(2500):
        om=rk4(rhs,om,0.01)
        if s%10==0:
            sub=om[aset]; traj.append(np.concatenate([sub.real,sub.imag]))
    X=np.array(traj); X=(X-X.mean(0))/X.std(0)
    Dbox=box_dim(X)
    print(f"nu={nu} A={A}: t={time.time()-t0:.0f}s n={n} active={len(aset)} box_dim={Dbox:.2f} E={0.5*np.sum(np.abs(om)**2):.2f}",flush=True)
