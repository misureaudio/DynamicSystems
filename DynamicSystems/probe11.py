"""Probe 11: Benettin (variational equation + Gram-Schmidt) on Poincare-disk geodesic flow.
Expect Lyapunov exponents {+1, 0, -1} for K=-1 (unit tangent bundle, 3-dim)."""
import numpy as np

def geodesic_rhs(x):
    px, py, vx, vy = x
    r2 = px*px + py*py
    one = 1.0 - r2
    v2 = vx*vx + vy*vy
    xv = px*vx + py*vy
    ax = (v2*px - 2.0*xv*vx)/one
    ay = (v2*py - 2.0*xv*vy)/one
    return np.array([vx, vy, ax, ay])

def geodesic_jac(x):
    # analytic Jacobian of geodesic_rhs
    px, py, vx, vy = x
    r2 = px*px + py*py
    one = 1.0 - r2
    v2 = vx*vx + vy*vy
    xv = px*vx + py*vy
    # ax = (v2*px - 2*xv*vx)/one ; ay = (v2*py - 2*xv*vy)/one
    d_one = 2.0*(px*vx + py*vy)  # d(one)/dx_i ... one=1-r2, d(one)= -2 x . but here we differentiate wrt state
    # Let A = v2*px - 2*xv*vx, ax = A/one
    # dA/dpx = 2*vx*px - 2*(vx*vx + xv) = 2*vx*px - 2*vx*vx - 2*xv
    # dA/dpy = 2*vx*py - 2*vy*xv
    # dA/dvx = 2*px*vx - 2*(xv + px*vx) = 2*px*vx - 2*xv - 2*px*vx = -2*xv
    # dA/dvy = -2*px*vy
    dApx = 2*vx*px - 2*vx*vx - 2*xv
    dApy = 2*vx*py - 2*vy*xv
    dAvx = -2*xv
    dAvy = -2*px*vy
    # one = 1 - px^2 - py^2 ; done/dpx = -2 px ; done/dpy = -2 py
    # ax = A/one -> dax/dx_i = (dA/dx_i * one - A * done/dx_i)/one^2
    A = v2*px - 2*xv*vx
    B = v2*py - 2*xv*vy
    one2 = one*one
    dax_dpx = (dApx*one - A*(-2*px))/one2
    dax_dpy = (dApy*one - A*(-2*py))/one2
    dax_dvx = (dAvx*one)/one2
    dax_dvy = (dAvy*one)/one2
    # ay = B/one
    dBpx = 2*vy*px - 2*vy*vx - 2*xv
    dBpy = 2*vy*py - 2*vx*xv
    dBvx = -2*xv
    dBvy = -2*py*vx
    day_dpx = (dBpx*one - B*(-2*px))/one2
    day_dpy = (dBpy*one - B*(-2*py))/one2
    day_dvx = (dBvx*one)/one2
    day_dvy = (dBvy*one)/one2
    J = np.zeros((4,4))
    J[0,2] = 1.0; J[1,3] = 1.0
    J[2,0] = dax_dpx; J[2,1] = dax_dpy; J[2,2] = dax_dvx; J[2,3] = dax_dvy
    J[3,0] = day_dpx; J[3,1] = day_dpy; J[3,2] = day_dvx; J[3,3] = day_dvy
    return J

def rk4_pair(f, J, x, V, dt):
    # integrate base x and tangent frame V (4x3) together
    def step(x, V, a, b):
        k1x = f(x); k1v = J(x) @ V
        k2x = f(x + 0.5*dt*k1x); k2v = J(x + 0.5*dt*k1x) @ (V + 0.5*dt*k1v)
        k3x = f(x + 0.5*dt*k2x); k3v = J(x + 0.5*dt*k2x) @ (V + 0.5*dt*k2v)
        k4x = f(x + dt*k3x); k4v = J(x + dt*k3x) @ (V + dt*k3v)
        xn = x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        Vn = V + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
        return xn, Vn
    return step(x, V, None, None)

def benettin_geodesic(x0, T=100.0, dt=0.01):
    x = x0.copy()
    V = np.eye(4)[:, :3]  # 4x3 tangent frame
    tot = np.zeros(3)
    nsteps = int(T/dt)
    for s in range(nsteps):
        x, V = rk4_pair(geodesic_rhs, geodesic_jac, x, V, dt)
        Q, R = np.linalg.qr(V)
        sgn = np.sign(np.diag(R)); sgn[sgn == 0] = 1
        tot += np.log(np.abs(np.diag(R)))
        V = Q * sgn
    return tot/(nsteps*dt)

px, py = 0.0, 0.2
r2 = px*px + py*py
vflat = (1.0 - r2)/2.0
x0 = np.array([px, py, vflat, 0.0])
lam = benettin_geodesic(x0)
print("geodesic flow Lyapunov exponents (K=-1):", np.round(lam, 4), " (expect {+1,0,-1})")

# also check the Jacobian against finite differences at a few points
rng = np.random.default_rng(0)
for trial in range(3):
    x = x0 + 0.3*rng.standard_normal(4)
    # keep inside disk
    x[:2] = x[:2]*0.3
    x[2:] = 0.5*x[2:]
    h = 1e-6
    Jnum = np.zeros((4,4)); f0 = geodesic_rhs(x)
    for j in range(4):
        xp = x.copy(); xp[j] += h
        Jnum[:, j] = (geodesic_rhs(xp) - f0)/h
    J = geodesic_jac(x)
    print(f"trial {trial}: max |J_an - J_num| = {np.abs(J - Jnum).max():.2e}")
