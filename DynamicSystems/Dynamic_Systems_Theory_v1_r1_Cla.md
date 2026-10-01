Let me look at what v1 actually says on these three points before responding.I'll test the Hénon claim directly instead of arguing from memory.On two of your three points the review stands. On the third we agree, and you adopted my correction.

## Hénon: the review stands, and I tested it

Your directions are right (unstable is near-horizontal, stable is steep), but the conclusion is reversed. I iterated the map at a=1.4, b=0.3 for 3 million points:

- **The x-projection is not a Cantor set.** It fills the whole interval [−1.28, 1.27], and every one of 2000 bins was occupied.
- **It is not a graph over x.** In a thin slice at x = 0.5, the y-values spread from −0.23 to 0.21, so there are many y-values per x.
- **The Cantor structure is in the steep y direction.** A vertical cross-section through a small box shows gaps. The fixed point's unstable eigenvector is (−0.988, 0.154), nearly horizontal, as you say.

The theory agrees. The attractor is the closure of the unstable manifold of the saddle fixed point, so it contains whole unstable curves, which are smooth. A better sentence for v1:

> The attractor is the closure of the unstable manifold of the saddle fixed point. Locally it is the product of a smooth arc along the nearly horizontal unstable direction and a Cantor set transverse to it, along the steep stable direction.

The last sentence of v1's Hénon paragraph, that the map was inspired by Lorenz and is not a literal Poincaré section, is good.

## Poincaré: I'd still change the name

"Poincaré's last geometric theorem" is the standard name for the Poincaré–Birkhoff theorem (Poincaré, 1912). I'm not aware of "last geometric problem" being the standard name for the persistence question. Poincaré's own phrase for that is the "fundamental problem of dynamics" (*Méthodes nouvelles*), and "conditionally periodic motions" is Kolmogorov's title. Some author may use the loose name, and your clarifying parenthetical softens the confusion, but I wouldn't call it standard. I'd retitle §4.2 "Poincaré's fundamental problem of dynamics and small divisors".

## SRB potential: we agree

v1 now has φ = −log|det Tf|_{E^u}, pressure zero, and the matching Gibbs statement. That is correct. The version I reviewed had +log in both §3.4 and §3.5, which is what I flagged, so this was a fix and not a disagreement.

## One new issue in v1

§4.3 says the surviving tori "have full Lebesgue measure." For fixed ε > 0 they have positive measure, with the destroyed set shrinking as ε → 0 (typically O(√ε)). It is the Diophantine frequencies of the *unperturbed* system that have full measure when you take the union over γ. The "Cantor type, not open" point right after it is correct.

I only re-read the passages you flagged. If you'd like, I can go through all of v1 against my earlier list.