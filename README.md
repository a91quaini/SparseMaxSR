
# SparseMaxSR.jl

Sparse maximum–Sharpe portfolio selection in Julia.  
This package provides utilities to:
- compute Sharpe ratios and mean–variance efficient (MVE) weights,
- select a **k-sparse** maximum–Sharpe portfolio either by exhaustive search or by a scalable **cutting-planes** algorithm inspired by Bertsimas & Cory‑Wright (2022).

It is solver-agnostic: you can plug in open-source solvers (HiGHS, GLPK, Clarabel, COSMO, SCS) or commercial solvers (CPLEX, Mosek, Gurobi).

---

## Features

- **SharpeRatio**:
  - `compute_sr(w, μ, Σ)` — Sharpe ratio of a given weight vector.
  - `compute_mve_sr(μ, Σ)` — maximum Sharpe ratio on a support.
  - `compute_mve_weights(μ, Σ)` — unconstrained MVE weights (Σ⁻¹μ).
- **MVESelection**:
  - `compute_mve_selection(μ, Σ, k; method=:auto)` — pick `k` assets to maximize Sharpe ratio.
    - `:exhaustive` search for small instances.
    - `:cutting_planes` outer-approximation for larger `n`.
- **CuttingPlanesUtils** (advanced):
  - building blocks used by the selection algorithm (dual QP, SOC/QCQP relaxations, Kelley cuts, warm starts).

Robust defaults:
- A tiny ridge `ε·mean(diag(Σ))·I` is applied by default (`ε = EPS_RIDGE[]`, defaults to `1e-6`).
- A **default conic optimizer** is selected at runtime (Clarabel → COSMO → SCS) unless you set it explicitly.

---

## Installation

```julia
pkg> add https://github.com/<your-org-or-user>/SparseMaxSR.jl
```

> If you’re developing locally:
>
> ```julia
> pkg> dev /path/to/SparseMaxSR
> pkg> activate /path/to/SparseMaxSR
> pkg> instantiate
> pkg> test
> ```

### Optional solvers

- **MILP master (cutting-planes)**: HiGHS.jl (open-source), GLPK.jl, Gurobi.jl, CPLEX.jl.
- **Conic/QP duals**: Clarabel.jl (open-source), COSMO.jl, SCS.jl, MosekTools.jl.

Install any you want, e.g.:
```julia
pkg> add HiGHS Clarabel
# or
pkg> add GLPK COSMO
# or
pkg> add CPLEX MosekTools
```

> **Licenses**: CPLEX/Gurobi/Mosek require separate installation and licenses. Make sure the solver works before using it in JuMP.

---

## Quickstart

```julia
using SparseMaxSR
using Random, LinearAlgebra

# small synthetic instance
Random.seed!(1)
n, k = 10, 3
μ = 0.05 .+ 0.10 .* rand(n)
A = randn(n,n); Σ = Symmetric(A*A' .+ 0.05I)  # positive definite-ish

# Compute MVE weights and SR
w = SparseMaxSR.compute_mve_weights(μ, Σ)      # unconstrained Σ⁻¹μ
sr = SparseMaxSR.compute_sr(w, μ, Σ)

# Select k assets (automatic method switching)
sel = SparseMaxSR.compute_mve_selection(μ, Σ, k)     # returns sorted indices of length k
```

### Choosing solvers

- **Default conic optimizer** (used inside dual subproblems):
  ```julia
  import SparseMaxSR: set_default_optimizer!
  import Clarabel
  set_default_optimizer!(() -> Clarabel.Optimizer())
  ```

- **MILP master** for the cutting-planes path:
  ```julia
  import HiGHS
  sel = SparseMaxSR.compute_mve_selection(μ, Σ, k;
      method = :cutting_planes,
      optimizer = () -> HiGHS.Optimizer())      # MILP master
  ```

### CPLEX / Mosek examples (only if installed)

```julia
# CPLEX as MILP master (cutting planes)
if Base.find_package("CPLEX") !== nothing
    import CPLEX
    sel = SparseMaxSR.compute_mve_selection(μ, Σ, k;
        method = :cutting_planes,
        optimizer = () -> CPLEX.Optimizer())
end

# Mosek for the dual QP/SOC relaxations
if Base.find_package("MosekTools") !== nothing
    import MosekTools
    sel = SparseMaxSR.compute_mve_selection(μ, Σ, k;
        method = :cutting_planes,
        optimizer = () -> HiGHS.Optimizer(),     # master
        dual_optimizer = () -> MosekTools.Optimizer())
end
```

---

## API Reference (selected)

### `compute_sr(weights, μ, Σ; selection=Int[], epsilon=EPS_RIDGE[], do_checks=false) -> Float64`

Sharpe ratio `SR = (w'μ) / √(w'Σw)`.  
If `selection` is given, the SR is computed on the corresponding subvector/submatrix. A small ridge is applied to Σ if `epsilon > 0` (defaults to `EPS_RIDGE[]`).

**Example**
```julia
w = ones(n) ./ n
sr = SparseMaxSR.compute_sr(w, μ, Σ)
```

### `compute_mve_sr(μ, Σ; selection=Int[], epsilon=EPS_RIDGE[], do_checks=false) -> Float64`

Maximum Sharpe ratio on a support using Σ⁻¹μ (Cholesky when SPD, pseudoinverse otherwise).

**Example**
```julia
sr_all = SparseMaxSR.compute_mve_sr(μ, Σ)
sr_on_sel = SparseMaxSR.compute_mve_sr(μ, Σ; selection = sel)
```

### `compute_mve_weights(μ, Σ; selection=Int[], γ=1.0, epsilon=EPS_RIDGE[], do_checks=false) -> Vector{Float64}`

Unconstrained MVE weights `w = (1/γ) Σ⁻¹μ`. Returned as a length-`n` vector, zeroed off-selection.

**Example**
```julia
w = SparseMaxSR.compute_mve_weights(μ, Σ; γ = 2.0)
```

### `compute_mve_selection(μ, Σ, k; method=:auto, exhaustive_threshold=20, exhaustive_max_combs=200_000, optimizer=nothing, attrs=NamedTuple(), dual_optimizer=nothing, dual_attrs=NamedTuple(), epsilon=EPS_RIDGE[], rng=Random.default_rng()) -> Vector{Int}`

Selects `k` assets that (approximately) maximize SR.

- `:exhaustive` — exact search for small instances.
- `:cutting_planes` — outer-approximation (MILP master + dual QP/SOC cuts).

**Example (automatic):**
```julia
sel = SparseMaxSR.compute_mve_selection(μ, Σ, 5)
```

**Example (cutting-planes with explicit solvers):**
```julia
import HiGHS, Clarabel
sel = SparseMaxSR.compute_mve_selection(μ, Σ, 10;
    method = :cutting_planes,
    optimizer = () -> HiGHS.Optimizer(),        # MILP
    dual_optimizer = () -> Clarabel.Optimizer() # duals
)
```

---

## Tuning & Tips

- **Ridge**: increase `EPS_RIDGE[]` if Σ is nearly singular or noisy:
  ```julia
  import SparseMaxSR: set_default_ridge!, EPS_RIDGE
  set_default_ridge!(1e-4)     # used by default in all Sharpe/MVE utilities
  ```
- **Exhaustive vs Cutting-planes**:
  - Exhaustive is exact but exponential in k.
  - Cutting-planes scales much better; pass fast solvers (HiGHS + Clarabel/COSMO/SCS/Mosek).
- **Reproducibility**: pass an `rng` to selection and warm-starts.
- **MILP attributes**: tune time limits and MIP gap via `attrs`, e.g.:
  ```julia
  attrs = (; "time_limit" => 120.0)   # or solver-specific keys
  sel = SparseMaxSR.compute_mve_selection(μ, Σ, k;
      method=:cutting_planes, optimizer=() -> HiGHS.Optimizer(), attrs=attrs)
  ```

---

## Testing

From the package root:
```julia
pkg> test
```

The test suite automatically **skips** solver-specific parts if the solver is not installed.  
Install any of HiGHS/GLPK/Clarabel/COSMO/SCS/CPLEX/MosekTools to enable more coverage.

---

## Troubleshooting

- **`No default optimizer set`**  
  Call `set_default_optimizer!(...)` or add Clarabel/COSMO/SCS to your environment.

- **`Package SparseMaxSR does not have CPLEX/MosekTools in its dependencies`**  
  We treat commercial solvers as **optional**. Add them to your project if you want to use them.

- **Numeric issues (NaNs, tiny negative variances)**  
  Increase ridge: `set_default_ridge!(1e-4)` or pass `epsilon=` in calls.

- **Slow MILP**  
  Try HiGHS (open source), tighten time limits/gaps, reduce `k`, or use better warm starts.

---

## References

- Bertsimas, D., & Cory‑Wright, R. (2022). *A scalable algorithm for sparse portfolio selection.*  
  (Kelley cutting-planes / in–out ideas inform our cutting-planes routine.)

- Markowitz, H. (1952). *Portfolio selection.* Journal of Finance.

---

## License

MIT © 2025 SparseMaxSR contributors

---

## Contributing

Issues and pull requests are welcome. Please include a small reproducer and your solver versions.
