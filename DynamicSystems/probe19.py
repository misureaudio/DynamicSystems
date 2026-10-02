"""Probe 19: Benettin on geodesic flow of H^2 via hyperboloid model.
ST M is 3-dim; embed in R^6 with 3 constraints. Df = [[0,I],[I,0]] (constant).
Evolve a 6x3 frame in the constraint tangent space, QR re-orthogonalize. Expect {+1,0,-1}."""
import numpy as np

ETA = np.diag([-1.0, 1.0, 1.0])
def mink(a, b): return float(np.dot(ETA @ a, b))

# constant Jacobian of the geodesic flow (x'=v, v'=x)
A = np.zeros((6, 6))
A[:3, 3:] = np.eye(3)
A[3:, :3] = np.eye(3)

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

def tangent_frame(x, seed):
    """6x3 orthonormal (Euclidean) frame spanning ker(grad_constraints) = T(ST M)."""
    rng = np.random.default_rng(seed)
    G = grad_constraints(x)
    U, S, Vt = np.linalg.svd(G)
    N = Vt[3:, :]            # 3x6: orthonormal basis of null space of G (Euclidean)
    return N.T               # 6x3

# initial point on ST M (hyperboloid), unit speed
X0 = np.array([np.cosh(0.5), np.sinh(0.5), 0.0])
V0 = np.array([np.sinh(0.5), np.cosh(0.5), 0.0])
x0 = np.concatenate([X0, V0])
print("constraints(x0):", constraints(x0))

# verify A preserves the constraint tangent space (so the frame stays in T(ST M))
V0f = tangent_frame(x0, 0)
resid = np.max(np.abs(constraints(np.concatenate([V0f[:, 0], np.zeros(3)]))))  # placeholder
# better: check A maps ker(G) into ker(G): G A v = 0 for v in ker(G)
G = grad_constraints(x0)
err = np.max(np.abs(G @ A @ V0f))
print("max |G A v| for v in frame (should be ~0):", err)

# Benettin: evolve frame V (6x3) via V' = A V (A constant), QR each step
V = tangent_frame(x0, 0)
tot = np.zeros(3)
dt = 0.01
T = 200.0
n = int(T / dt)
for i in range(n):
    V = V + dt * (A @ V)
    Q, R = np.linalg.qr(V)
    sgn = np.sign(np.diag(R)); sgn[sgn == 0] = 1
    tot += np.log(np.abs(np.diag(R)))
    V = Q * sgn
lam = np.sort(tot / T)[::-1]
print("Benettin Lyapunov exponents (K=-1):", np.round(lam, 4), " (expect {+1, 0, -1})")

# cross-check: eigenvalues of A
eigs = np.linalg.eigvals(A)
print("eigenvalues of A:", np.round(np.sort(np.real(eigs)), 4))
print("distinct real parts (the 3 exponents):", np.round(np.sort(np.unique(np.round(np.real(eigs), 6)))[::-1], 4))

# also: the frame should stay in the constraint tangent space
Gend = grad_constraints(x0)  # A constant so x0 fixed for the linearized dynamics check
print("frame still in ker(G) after evolution: max|G V| =", np.max(np.abs(Gend @ V)))
