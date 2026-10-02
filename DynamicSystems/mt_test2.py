import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
tests = [
 r'$\frac{1}{n}\log\|Tf^n v\|$',
 r'time average $\frac{1}{n}\sum g(f^k x)$',
 r'Geodesic on $\mathbb H^2$ (Poincare disk, $K=-1$)',
 r'$\theta$',
 r'time average of $x_1^2$',
 r'space average $\int g\,d\mathrm{Leb}=1/3$',
 r'$\frac{1}{2}$',
 r'$\omega(I)=\partial H_0/\partial I$',
 r'$\varphi=-\log|Tf|_{E^u}$',
 r'$\mathcal P$',
 r'$\bigvee_{k=0}^{n-1}$',
 r'$\mathrm{Leb}$',
]
bad = 0
for t in tests:
    fig, ax = plt.subplots()
    try:
        ax.set_ylabel(t)
        ax.set_title(t)
        fig.canvas.draw()
        plt.close(fig)
        print("OK  ", t)
    except Exception as e:
        bad += 1
        plt.close(fig)
        print("FAIL", t, " ->", str(e)[:90])
print("failures:", bad)
