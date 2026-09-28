# Baselines

`methods.py` provides controlled adapters for DiCE-style genetic search, FACE (`k=10` graph), Growing Spheres (`r0=0.10`, `delta=0.10`, 1000 samples/layer), and LORELEY-style constrained evolution (`N=1000`, `G=10`, `pc=0.80`, `pm=0.20`). They share the predictor, test prefixes, target selection, domains, and feasibility repair used by SAGE.

The paper does not publish the authors' baseline source code or every validation-selected setting. These are transparent reimplementations, not a claim that the unavailable original baseline code has been recovered.
