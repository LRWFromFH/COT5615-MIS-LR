# COT5615-MIS-LR
COT5615 Math for Intelligent Systems group project: Convex Classification Beyond Softmax.

**Report (LaTeX, Overleaf):** https://www.overleaf.com/project/6afe07eec20f8dc40c762249/share#7aed1e41707595ae8dd7cd25a667fa1bb4fc6e58554a0b70

## Part 1: Binary classification

Binary classifiers trained by minimizing a matching loss built from a convex potential, with the bias left out of the
L2 penalty:

```
E(w) = sum_n [phi*(u_n) - y_n u_n] + (lam/2) w^T C w,   u = Xw,   C = diag(1, ..., 1, 0)
g    = X^T (sigma(u) - y) + lam C w
H v  = X^T (sigma'(u) * (X v)) + lam C v     (H is never formed)
```

Two activations:
- **Sigmoid** (logistic regression): `phi*` is softplus.
- **Arctan**: `sigma(u) = 1/2 + arctan(u)/pi`. The spec's integral of sigma from -inf diverges for this one, so `phi*` is
  the integral from 0 instead (the Legendre conjugate). Its loss has no floor, so the objective comes out negative and
  `||w||` grows like `lam^(-1/2)`.

### Files
- `src/binary.py`: the activations (with numerically stable forms, no clipping) and the `BinaryObjective` class with
  `objective`, `gradient_and_curvature` and `hessian_vector_product`, plus call counters. Also `log_loss` and `accuracy`.
- `src/data.py`: `load_task("digits01" | "breast_cancer")`. 75/25 stratified split (seed 0), standardized with
  training statistics only, bias column of ones appended last.
- `notebooks/part1_binary.ipynb`: the checks, for both datasets, both activations and lam in {0.01, 0.1, 1, 10}:
  1. stability of the naive vs stable formulas at extreme inputs
  2. gradient and HVP vs central finite differences (relative error about 1e-10 to 1e-9)
  3. convexity: Hessian built from HVPs is symmetric with a positive smallest eigenvalue
  4. fitting with SciPy `trust-ncg` using our gradient and HVP; the sigmoid fit matches scikit-learn

### Running
Needs numpy, scipy, pandas and scikit-learn. Open `notebooks/part1_binary.ipynb` and run all cells; it finds `src/` on
its own. In PyCharm, mark `src` as a Sources Root so the imports resolve.

## references/
The two notebooks that were originally at the top level (`Linear-Regression.ipynb`, `Logistic-Regression.ipynb`). They
use a separate figure dataset and are not part of the project; kept only for reference.
