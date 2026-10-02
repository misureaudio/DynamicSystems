"""Probe 20: Benettin on geodesic flow of H^2 (hyperboloid model), Sasaki-metric Gram-Schmidt.
Flow: x' = (V, X) on ST M (3-dim). Variational: V' = A V, A = [[0,I],[I,0]] (constant).
Re-orthogonalize the frame in the moving tangent space ker G(x(t)) using the Sasaki metric.
Expect Lyapunov exponents {+1, 0, -1}."""
import numpy as np

ETA = np.diag([-1.0, 1.0, 1.0])
def mink(a, b): return float(np.dot(ETA @ a, b))

A = np.zeros((6, 6))
A[:3, 3:] = np.eye(3)
A[3:, :3] = np.eye(3)

def F(x):
    X, V = x[:3], x[3:]
    return np.concatenate([V, X])  # v.v = 1

def constraints(x):
    X, V = x[:3], x[3:]
    return np.array([mink(X, X) + 1.0, mink(X, V), mink(V, V) - 1.0])

def grad_constraints(x):
    X, V = x[:3], x[3:]
    G = np.zeros((3, 6))
    G[0, :3] = 2 * (ETA @ X)
    G[1, :3] = ETA @ V; G[1, 3:] = ETA @ X
    G[2, 3:] = 2 * (ETA @ V)
    return G

def sasaki(v1, v2):
    w1, z1 = v1[:3], v1[3:]
    w2, z2 = v2[:3], v2[3:]
    return mink(w1, w2) + mink(z1, z2)

def sasaki_gs(V, x):
    """Gram-Schmidt of 6x3 frame V (columns in T_x(ST M)) w.r.t. Sasaki metric.
    Returns (Q, diag_factors)."""
    Q = np.zeros_like(V)
    diag = np.zeros(3)
    for i in range(3):
        u = V[:, i].copy()
        for j in range(i):
            u -= sasaki(u, Q[:, j]) / sasaki(Q[:, j], Q[:, j]) * Q[:, j]
        # verify u stays in tangent space
        g = sasaki(u, u)
        assert g > 0, f"degenerate at col {i}"
        diag[i] = np.sqrt(g)
        Q[:, i] = u / np.sqrt(g)
    return Q, diag

def rk4_base(x, dt):
    k1 = F(x); k2 = F(x + 0.5*dt*k1); k3 = F(x + 0.5*dt*k2); k4 = F(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rk4_frame(V, dt):
    k1 = A @ V; k2 = A @ (V + 0.5*dt*k1); k3 = A @ (V + 0.5*dt*k2); k4 = A @ (V + dt*k3)
    return V + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# initial point on ST M
X0 = np.array([np.cosh(0.5), np.sinh(0.5), 0.0])
V0 = np.array([np.sinh(0.5), np.cosh(0.5), 0.0])
x0 = np.concatenate([X0, V0])
print("constraints(x0):", np.max(np.abs(constraints(x0))))

# initial frame: null space of G(x0), then Sasaki-GS
G = grad_constraints(x0)
U, S, Vt = np.linalg.svd(G)
V = Vt[3:, :].T.copy()          # 6x3, in ker G(x0)
V, _ = sasaki_gs(V, x0)

dt = 0.01
T = 300.0
reorth = 20  # every 0.2 s
tot = np.zeros(3)
n = int(T / dt)
x = x0.copy()
for i in range(1, n + 1):
    x = rk4_base(x, dt)
    V = rk4_frame(V, dt)
    if i % reorth == 0:
        V, diag = sasaki_gs(V, x)
        tot += np.log(diag)
        # frame should stay in the moving tangent space
        err = np.max(np.abs(grad_constraints(x) @ V))
        if err > 1e-8:
            print(f"WARNING: frame left tangent space at step {i}, err={err:.2e}")
lam = np.sort(tot / T)[::-1]
print(f"Benettin exponents (K=-1, T={T}):", np.round(lam, 4), " (expect {+1, 0, -1})")
print("constraint drift max:", np.max(np.abs(constraints(x))))

# convergence in T
for TT in [50, 100, 300]:
    x = x0.copy(); V = Vt[3:, :].T.copy(); V, _ = sasaki_gs(V, x0)
    tot = np.zeros(3)
    for i in range(1, int(TT/dt) + 1):
        x = rk4_base(x, dt); V = rk4_frame(V, dt)
        if i % reorth == 0:
            V, diag = sasaki_gs(V, x); tot += np.log(diag)
    print(f"  T={TT}: {np.round(np.sort(tot/TT)[::-1], 4)}")
