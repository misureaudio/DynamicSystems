"""Probe 21: Benettin on geodesic flow of H^2 (hyperboloid model).
Flow (x,v)' = (v, x) on ST M (3-dim). Variational V' = A V, A=[[0,I],[I,0]] (constant).
Re-orthogonalize the frame in the moving tangent space ker G(x) using the EUCLIDEAN metric
(positive definite on the tangent subspace). Expect Lyapunov exponents {+1, 0, -1}."""
import numpy as np

ETA = np.diag([-1.0, 1.0, 1.0])
def mink(a, b): return float(np.dot(ETA @ a, b))

A = np.zeros((6, 6))
A[:3, 3:] = np.eye(3)
A[3:, :3] = np.eye(3)

def F(x):
    X, V = x[:3], x[3:]
    return np.concatenate([V, X])

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

def eucl_gs(V):
    Q, R = np.linalg.qr(V)
    sgn = np.sign(np.diag(R)); sgn[sgn == 0] = 1
    return Q * sgn, np.abs(np.diag(R))

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

# initial frame: null space of G(x0)
G = grad_constraints(x0)
U, S, Vt = np.linalg.svd(G)
V = Vt[3:, :].T.copy()          # 6x3, in ker G(x0)
V, _ = eucl_gs(V)

def benettin(T, reorth=20, dt=0.01):
    x = x0.copy()
    G0 = grad_constraints(x0)
    U, S, Vt = np.linalg.svd(G0)
    V = Vt[3:, :].T.copy(); V, _ = eucl_gs(V)
    tot = np.zeros(3)
    n = int(T / dt)
    for i in range(1, n + 1):
        x = rk4_base(x, dt)
        V = rk4_frame(V, dt)
        if i % reorth == 0:
            V, diag = eucl_gs(V)
            tot += np.log(diag)
    return np.sort(tot / T)[::-1]

for TT in [50, 100, 300]:
    print(f"T={TT}:", np.round(benettin(TT), 4), " (expect {+1, 0, -1})")

# check frame stays in the moving tangent space
x = x0.copy(); G0 = grad_constraints(x0)
U, S, Vt = np.linalg.svd(G0); V = Vt[3:, :].T.copy(); V, _ = eucl_gs(V)
dt = 0.01
worst = 0.0
for i in range(1, 1001):
    x = rk4_base(x, dt); V = rk4_frame(V, dt)
    if i % 20 == 0:
        V, _ = eucl_gs(V)
        worst = max(worst, np.max(np.abs(grad_constraints(x) @ V)))
print("max |G(x) V| over T=10 (frame in tangent space):", worst)

# cross-check via eigenvalues of the linearized operator
eigs = np.linalg.eigvals(A)
print("eigenvalues of A (6x6):", np.round(np.sort(np.real(eigs)), 4))
print("  -> distinct real parts {+1, -1}; the 0 exponent is the flow direction")
