import matplotlib
matplotlib.use("Agg")
from matplotlib.mathtext import MathTextParser
tests = [
 r'$\frac{1}{n}\log\|Tf^n v\|$',
 r'time average $\frac{1}{n}\sum g(f^k x)$',
 r'Geodesic on $\mathbb H^2$ (Poincare disk, $K=-1$)',
 r'$\theta$',
 r'time average of $x_1^2$',
 r'space average $\int g\,d\mathrm{Leb}=1/3$',
 r'$\frac{1}{2}$',
 r'$\omega(I)=\partial H_0/\partial I$',
]
p = MathTextParser("path")
bad = 0
for t in tests:
    try:
        p.parse(t, 0, 10, 72)
        print("OK  ", t)
    except Exception as e:
        bad += 1
        print("FAIL", t, " ->", str(e)[:90])
print("failures:", bad)
