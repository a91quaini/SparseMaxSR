# SparseMaxSR.jl

SparseMaxSR.jl provides fast, reproducible utilities for **sparse maximum–Sharpe** portfolio selection in Julia. It includes:

- **Sharpe & MVE utilities**
  - `compute_sr(w, μ, Σ)` — Sharpe ratio of given weights.
  - `compute_mve_sr(μ, Σ; selection=:)` — Maximum Sharpe on a support.
  - `compute_mve_weights(μ, Σ; selection=:)` — Mean–variance efficient (MVE) weights.
- **k‑sparse selection**
  - `compute_mve_selection(μ, Σ, k; method=:exhaustive or :cutting_planes)` — choose `k` assets to maximize Sharpe.  
    Cutting‑planes scales to larger problems; exhaustive is exact for small ones.
- **Solver‑friendly design**
  - Works with free solvers (SCS, Clarabel, COSMO, HiGHS, GLPK) and commercial ones (CPLEX, Mosek, Gurobi).
  - A **default conic/QP solver** is auto‑detected and can be **persisted** across sessions via `Preferences.jl`.

---

## Installation

Add the package (replace URL if you host under a different org/user):

```julia
pkg> add https://github.com/<your-user-or-org>/SparseMaxSR.jl
```

If developing locally:

```julia
pkg> dev /path/to/SparseMaxSR
pkg> activate /path/to/SparseMaxSR
pkg> instantiate
pkg> test
```

### Optional solvers

Install any of the following as needed:

```julia
pkg> add SCS Clarabel COSMO HiGHS GLPK
# (commercial) pkg> add CPLEX MosekTools Gurobi
```

> Commercial solvers require external installation/licensing.

---

## Quickstart

```julia
using SparseMaxSR
using Random, LinearAlgebra

# synthetic instance
Random.seed!(1)
n, k = 12, 4
μ = 0.03 .+ 0.12 .* rand(n)

A = randn(n,n)
Σ = Symmetric(A*A' .+ 0.05I)  # SPD-ish

# Unconstrained MVE weights and Sharpe
w  = compute_mve_weights(μ, Σ)      # returns Vector{Float64}
sr = compute_sr(w, μ, Σ)

# k-sparse selection (auto-chooses a method & conic solver)
sel = compute_mve_selection(μ, Σ, k)  # Vector{Int} of length k

# Maximum Sharpe restricted to the selected support
sr_k = compute_mve_sr(μ, Σ; selection=sel)
```

**Tip:** If Σ is ill‑conditioned/noisy, increase the global ridge:

```julia
import SparseMaxSR: set_default_ridge!
set_default_ridge!(1e-4)
```

---

## Choosing and persisting solvers

SparseMaxSR separates **two** solver roles in the cutting‑planes method:

1. **Conic/QP dual solver** (used for dual relaxations): SCS / Clarabel / COSMO / Mosek
2. **MILP master solver** (for the master problem): HiGHS / GLPK / CPLEX / Gurobi / Mosek

You can:

### A) Persist your preferred conic/QP solver across sessions

```julia
# Once per environment (saved via Preferences.jl)
using SparseMaxSR
save_default_solver!("Clarabel")   # or "SCS", "COSMO", "auto"
```

### B) Override the conic/QP solver for the current session only

```julia
using SparseMaxSR, Clarabel
set_default_optimizer!(() -> Clarabel.Optimizer())
```

### C) Specify explicit solvers per call (recommended for cutting‑planes)

```julia
using HiGHS, Clarabel
sel = compute_mve_selection(μ, Σ, 10;
    method         = :cutting_planes,
    optimizer      = () -> HiGHS.Optimizer(),      # MILP master
    dual_optimizer = () -> Clarabel.Optimizer()    # conic/QP duals
)
```

> If you don’t set anything, the conic/QP dual solver is **auto‑detected**
> in this order: **SCS → Clarabel → COSMO**. You must pass a MILP master
> when using `method=:cutting_planes`.

---

## API Sketch

### `compute_sr(w, μ, Σ; selection=Int[], epsilon=EPS_RIDGE[], do_checks=false) -> Float64`
Sharpe ratio `SR = (w'μ) / sqrt(w'Σw)`. If `selection` is provided, the SR is computed on that subvector/submatrix. A ridge `epsilon` (default `EPS_RIDGE[]`) can stabilize Σ.

### `compute_mve_weights(μ, Σ; selection=Int[], γ=1.0, epsilon=EPS_RIDGE[], do_checks=false) -> Vector{Float64}`
Unconstrained MVE weights `w = (1/γ) Σ⁻¹ μ`. Zeros off‑selection. Uses Cholesky/pinv as appropriate.

### `compute_mve_sr(μ, Σ; selection=Int[], epsilon=EPS_RIDGE[], do_checks=false) -> Float64`
Maximum Sharpe on a support computed via MVE weights.

### `compute_mve_selection(μ, Σ, k; method=:auto, exhaustive_threshold=20, exhaustive_max_combs=200_000, optimizer=nothing, attrs=NamedTuple(), dual_optimizer=nothing, dual_attrs=NamedTuple(), epsilon=EPS_RIDGE[], rng=Random.default_rng()) -> Vector{Int}`
Choose `k` assets to (approximately) maximize Sharpe.  
- `:exhaustive` is exact for small problems.  
- `:cutting_planes` scales to larger `n` and `k`.  
- Pass `optimizer` for the **MILP** master; `dual_optimizer` for the **conic/QP** subproblems.  
- Use `attrs`/`dual_attrs` to pass solver parameters (via `MOI.RawParameter`).

---

## Configuration & tips

- **Ridge:** Set once globally with `set_default_ridge!(ε)`; or override per call via `epsilon=...`.
- **Reproducibility:** Many routines accept `rng` — pass your own `MersenneTwister` to fix randomness.
- **Time limits / gaps (MILP):** Use `attrs = (; "time_limit" => 120.0, "mip_gap" => 0.01)` (keys are solver‑specific).

---

## Testing

From the package root:

```julia
pkg> test
```

The suite skips solver‑specific tests when a solver is not installed. Install HiGHS/GLPK/Clarabel/COSMO/SCS/CPLEX/MosekTools to enable more coverage.

---

## Citing

If this package is useful in academic work, please cite Markowitz (1952) and algorithmic references on cutting‑planes for sparse portfolio selection (e.g., Kelley cutting‑planes / in‑out methods).

---

## License

MIT © 2025 SparseMaxSR contributors
