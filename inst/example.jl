
#!/usr/bin/env julia
# Extended example: compare Exhaustive, Lasso (path), and MIQP(heuristic)
# - Experiment A:  T=500, N=30,   k ∈ {1,3,5,7,9}   (Exhaustive + Lasso + MIQP)
# - Experiment B:  T=500, N=100,  k ∈ {1,5,10,15,20} (Lasso + MIQP)
#
# Run:  julia --project=. example.jl
#
# Notes:
# * Exhaustive search is skipped automatically when C(N,k) exceeds a cap.
# * Cells show "SR / time_s", where SR is the Sharpe ratio and time_s is seconds.

using SparseMaxSR
using Random, LinearAlgebra, Statistics, Printf, Dates
using Combinatorics: binomial

# --- status tracking helpers for summaries ---
import MathOptInterface as MOI  # for checking MIQP optimality symbol

# lightweight pretty-printer for a set of k's
function _fmt_ks(v::Vector{Int})
    isempty(v) && return "(none)"
    return join(sort(unique(v)), ", ")
end

# -----------------------
# Helpers
# -----------------------

# Pretty date stamp
timestamp() = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")

# Simulate returns with a mild 2-factor structure + noise
function simulate_returns(T::Int, N::Int; nf::Int=2, beta_scale=0.3, eps_scale=0.7, rng = Random.default_rng())
    F = randn(rng, T, nf)
    B = beta_scale .* randn(rng, N, nf)
    E = eps_scale  .* randn(rng, T, N)
    R = F * B' .+ E
    return R
end

# Compute μ and Σ from returns
means_and_cov(R) = (vec(mean(R, dims=1)), cov(R; corrected=true))

# Unified cell formatter: "SR / time_s"
cell(sr, t) = isnan(sr) ? "-" : @sprintf("%.4f / %.2fs", sr, t)

# Exhaustive: signature (μ, Σ, k; exactly_k=true, ...)
function run_exhaustive(μ, Σ, k)
    sel = w = nothing; sr = NaN; st = :UNKNOWN
    tsec = @elapsed begin
        sel, w, sr, st = SparseMaxSR.mve_exhaustive_search(μ, Σ, k; exactly_k=true)
    end
    return sel, w, sr, tsec, st
end

# MIQP heuristic: signature (μ, Σ; k=...)
function run_miqp_heuristic(μ, Σ, k)
    sel = w = nothing; sr = NaN; st = :UNKNOWN
    tsec = @elapsed begin
        sel, w, sr, st = SparseMaxSR.mve_miqp_heuristic_search(μ, Σ; k=k)
    end
    return sel, w, sr, tsec, st
end

# LASSO (path, no tuning): current package signature used here is (μ, Σ, T; ...)
function run_lasso_path(R, μ, Σ, k, alpha)
    sel = w = nothing; sr = NaN; st = :UNKNOWN
    tsec = @elapsed begin
        sel, w, sr, st = SparseMaxSR.mve_lasso_relaxation_search(
            μ, Σ, size(R,1);
            k = k,
            nlambda = 100,
            lambda_min_ratio = 1e-5,
            alpha = alpha,
            standardize = false,
            epsilon = SparseMaxSR.EPS_RIDGE,
            stabilize_Σ = true,
            do_checks = false
        )
    end
    return sel, w, sr, tsec, st
end

# Print a compact table: rows are k, columns are methods; cells "SR / time_s"
function print_table(title::AbstractString, ks::Vector{Int}, methods::Vector{String}, cells::Dict{Tuple{Int,String},String})
    println()
    println(title)
    println("-"^max(10, length(title)))
    # header
    @printf("%-6s", "k")
    for m in methods
        @printf(" | %-18s", m)
    end
    println()
    println("-"^(6 + length(methods)*(3+18)))
    # rows
    for k in ks
        @printf("%-6d", k)
        for m in methods
            c = get(cells, (k,m), "-")
            @printf(" | %-18s", c)
        end
        println()
    end
    println()
end

# -----------------------
# Experiment A: T=500, N=30, k in 1,3,5,7,9
# -----------------------
Random.seed!(42)
T, N = 500, 30
ks = [1,3,5,7,9]
methods = ["EXHAUSTIVE", "LASSO-PATH", "MIQP-HEURISTIC"]
cells = Dict{Tuple{Int,String},String}()
# trackers for summaries
lasso_almost_B = Int[]
miqp_notopt_B = Int[]
# trackers for summaries
lasso_almost_A = Int[]
miqp_notopt_A = Int[]

println("SparseMaxSR example — $(timestamp())")
println("Experiment A: T=$T, N=$N; methods=$(join(methods, ", "))")
R = simulate_returns(T, N)
μ, Σ = means_and_cov(R)

# Exhaustive safety: skip if too many combinations
EXH_CAP = 3_000_000  # allow up to ~3e6 subsets (30 choose 7 ≈ 2,035,800; 30 choose 9 ≈ 14,307,150 -> skipped)

for k in ks
    # Exhaustive (guarded)
    if binomial(N,k) <= EXH_CAP
        try
            _, _, sr, t, _ = run_exhaustive(μ, Σ, k)
            cells[(k,"EXHAUSTIVE")] = cell(sr, t)
        catch err
            cells[(k,"EXHAUSTIVE")] = "ERR"
        end
    else
        cells[(k,"EXHAUSTIVE")] = "SKIP"
    end

    # LASSO path
    try
        _, _, sr, t, st = run_lasso_path(R, μ, Σ, k, 0.99)
        cells[(k,"LASSO-PATH")] = cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K || st == :LASSO_RELAXATION_EMPTY
            push!(lasso_almost_A, k)
        end
    catch err
        cells[(k,"LASSO-PATH")] = "ERR"
    end

    # MIQP heuristic
    try
        _, _, sr, t, st = run_miqp_heuristic(μ, Σ, k)
        cells[(k,"MIQP-HEURISTIC")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_A, k)
        end
    catch err
        cells[(k,"MIQP-HEURISTIC")] = "ERR"
    end
end

print_table("Results — Experiment A (T=500, N=30)", ks, methods, cells)
println("LASSO: support size < k for k ∈ {" * _fmt_ks(lasso_almost_A) * "}")
println("MIQP: solver not OPTIMAL for k ∈ {" * _fmt_ks(miqp_notopt_A) * "}")


# -----------------------
# Experiment B: T=500, N=100, k in 1,5,10,15,20
# -----------------------

Random.seed!(1729)  # different seed
T, N = 500, 100
ks = [1; collect(5:5:70)]
methods = ["LASSO-PATH", "MIQP-HEURISTIC"]
cells = Dict{Tuple{Int,String},String}()
# trackers for summaries
lasso_almost_A = Int[]
miqp_notopt_A = Int[]

println("Experiment B: T=$T, N=$N; methods=$(join(methods, ", "))")
R = simulate_returns(T, N)
μ, Σ = means_and_cov(R)

for k in ks
    # LASSO path
    try
        _, _, sr, t, st = run_lasso_path(R, μ, Σ, k, 0.95)
        cells[(k,"LASSO-PATH")] = cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K || st == :LASSO_RELAXATION_EMPTY
            push!(lasso_almost_A, k)
        end
    catch err
        cells[(k,"LASSO-PATH")] = "ERR"
    end

    # MIQP heuristic
    try
        _, _, sr, t, st = run_miqp_heuristic(μ, Σ, k)
        cells[(k,"MIQP-HEURISTIC")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_A, k)
        end
    catch err
        cells[(k,"MIQP-HEURISTIC")] = "ERR"
    end
end

print_table("Results — Experiment B (T=500, N=100)", ks, methods, cells)
println("LASSO: support size < k for k ∈ {" * _fmt_ks(lasso_almost_B) * "}")
println("MIQP: solver not OPTIMAL for k ∈ {" * _fmt_ks(miqp_notopt_B) * "}")

# ------------------------------------------
# Experiment C: T=250*5, N=500, k in 100,500
# ------------------------------------------

Random.seed!(2025)
T, N = 250*5, 500
ks = [100, 500]
methods = ["LASSO-PATH", "MIQP-HEURISTIC"]
cells = Dict{Tuple{Int,String},String}()
lasso_almost_C = Int[]
miqp_notopt_C = Int[]

println("Experiment C: T=$T, N=$N; methods=$(join(methods, ", "))")
R = simulate_returns(T, N)
μ, Σ = means_and_cov(R)

for k in ks
    # LASSO path
    try
        _, _, sr, t, st = run_lasso_path(R, μ, Σ, k, 0.95)
        cells[(k,"LASSO-PATH")] = cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K || st == :LASSO_RELAXATION_EMPTY
            push!(lasso_almost_C, k)
        end
    catch err
        cells[(k,"LASSO-PATH")] = "ERR"
    end

    # MIQP heuristic
    try
        _, _, sr, t, st = run_miqp_heuristic(μ, Σ, k)
        cells[(k,"MIQP-HEURISTIC")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_C, k)
        end
    catch err
        cells[(k,"MIQP-HEURISTIC")] = "ERR"
    end
end

print_table("Results — Experiment C (T=250*2, N=500)", ks, methods, cells)
println("LASSO: support size < k for k ∈ {" * _fmt_ks(lasso_almost_C) * "}")
println("MIQP: solver not OPTIMAL for k ∈ {" * _fmt_ks(miqp_notopt_C) * "}")

# ------------------------------------------
# Experiment D: T=250*2, N=1000, k in 100
# ------------------------------------------

Random.seed!(2025)
T, N = 250*2, 1000
ks = [500]
methods = ["LASSO-PATH", "MIQP-HEURISTIC"]
cells = Dict{Tuple{Int,String},String}()
lasso_almost_D = Int[]
miqp_notopt_D = Int[]

println("Experiment D: T=$T, N=$N; methods=$(join(methods, ", "))")
R = simulate_returns(T, N)
μ, Σ = means_and_cov(R)

for k in ks
    # LASSO path
    try
        _, _, sr, t, st = run_lasso_path(R, μ, Σ, k, 0.7)
        cells[(k,"LASSO-PATH")] = cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K || st == :LASSO_RELAXATION_EMPTY
            push!(lasso_almost_D, k)
        end
    catch err
        cells[(k,"LASSO-PATH")] = "ERR"
    end

    # MIQP heuristic
    try
        _, _, sr, t, st = run_miqp_heuristic(μ, Σ, k)
        cells[(k,"MIQP-HEURISTIC")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_D, k)
        end
    catch err
        cells[(k,"MIQP-HEURISTIC")] = "ERR"
    end
end

print_table("Results — Experiment D (T=250*2, N=1000)", ks, methods, cells)
println("LASSO: support size < k for k ∈ {" * _fmt_ks(lasso_almost_D) * "}")
println("MIQP: solver not OPTIMAL for k ∈ {" * _fmt_ks(miqp_notopt_D) * "}")