#!/usr/bin/env julia
# Example: compare Exhaustive, LASSO (vanilla/refit with α-grid CV), and MIQP (vanilla/refit)
# - Single experiment to keep the script fast and aligned to current API
# - LASSO uses α-grid: 0.05:0.05:0.95 with forward-rolling OOS-CV (requires R)
#
# Run:
#   julia --project=. example.jl
#
using SparseMaxSR
using Random, LinearAlgebra, Statistics, Printf, Dates
using Combinatorics: binomial
import MathOptInterface as MOI

timestamp() = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

# -----------------------
# Simulation helpers
# -----------------------
function simulate_returns(T::Int, N::Int; nf::Int=2, beta_scale=0.3, eps_scale=0.7, rng = Random.default_rng())
    F = randn(rng, T, nf)
    B = beta_scale .* randn(rng, N, nf)
    E = eps_scale  .* randn(rng, T, N)
    return F * B' .+ E
end

means_and_cov(R) = (vec(mean(R, dims=1)), cov(R; corrected=true))
cell(sr, t) = isnan(sr) ? "-" : @sprintf("%.4f / %.2fs", sr, t)
_fmt_ks(v::Vector{Int}) = isempty(v) ? "(none)" : join(sort(unique(v)), ", ")

# -----------------------
# Runners (aligned to current API)
# -----------------------

# MIQP heuristic — REFIT (refit exact MVE on MIQP-selected support)
function run_miqp_refit(μ, Σ, k)
    tsec = 0.0
    res = nothing
    tsec = @elapsed begin
        res = SparseMaxSR.mve_miqp_heuristic_search(
            μ, Σ;
            k = k,
            m = max(0, k-1),
            exactly_k = false,
            fmin = zeros(length(μ)),
            fmax = fill(0.5, length(μ)),
            expand_rounds = 5,
            expand_factor = 2.0,
            expand_tol    = 5e-2,
            mipgap        = 1e-4,
            time_limit    = 60.0,
            threads       = max(Threads.nthreads()-1, 1),
            epsilon       = SparseMaxSR.EPS_RIDGE,
            stabilize_Σ   = true,
            normalize_weights  = false,
            compute_weights = true,
            use_refit     = true,
            do_checks     = false
        )
    end
    return res.selection, res.weights, res.sr, res.status, tsec
end

# LASSO-REFIT with α-grid CV (moment-only entry, requires R)
function run_lasso_refit_grid(R, μ, Σ, k; agrid = collect(0.05:0.05:0.95), cv_folds::Int=5)
    tsec = 0.0
    res = nothing
    tsec = @elapsed begin
        res = SparseMaxSR.mve_lasso_relaxation_search(
            μ, Σ, size(R,1);
            R = R,                     # enables α-grid CV
            k = k,
            nlambda = 100,
            lambda_min_ratio = 1e-3,
            alpha = agrid,             # GRID → CV
            standardize = false,
            epsilon = SparseMaxSR.EPS_RIDGE,
            stabilize_Σ = true,
            compute_weights = true,    # request refit weights
            normalize_weights = false,
            use_refit = true,          # REFIT (exact MVE on support)
            do_checks = false,
            cv_folds = cv_folds,
            cv_verbose = false
        )
    end
    return res.selection, res.weights, res.sr, res.status, res.alpha, tsec
end

# LASSO-REFIT with α-grid CV (moment-only entry, requires R)
function run_lasso_refit(R, μ, Σ, k; alpha = 0.75)
    tsec = 0.0
    res = nothing
    tsec = @elapsed begin
        res = SparseMaxSR.mve_lasso_relaxation_search(
            μ, Σ, size(R,1);
            R = R,                     # enables α-grid CV
            k = k,
            nlambda = 100,
            lambda_min_ratio = 1e-3,
            alpha = alpha,             # GRID → CV
            standardize = false,
            epsilon = SparseMaxSR.EPS_RIDGE,
            stabilize_Σ = true,
            compute_weights = true,    # request refit weights
            normalize_weights = false,
            use_refit = true,          # REFIT (exact MVE on support)
            do_checks = false
        )
    end
    return res.selection, res.weights, res.sr, res.status, res.alpha, tsec
end

# Pretty table
function print_table(title::AbstractString, ks::Vector{Int}, methods::Vector{String}, cells::Dict{Tuple{Int,String},String})
    println()
    println(title)
    println("-"^max(10, length(title)))
    @printf("%-6s", "k")
    for m in methods
        @printf(" | %-22s", m)
    end
    println()
    println("-"^(6 + length(methods)*(3+22)))
    for k in ks
        @printf("%-6d", k)
        for m in methods
            c = get(cells, (k,m), "-")
            @printf(" | %-22s", c)
        end
        println()
    end
    println()
end

# =============================================================================
# Experiment A: T=500, N=30, k in 1,3,5,7,9
# =============================================================================
Random.seed!(42)
T, N = 20, 30
ks = [5, 10] # collect(1:1:15)
methods = ["LASSO-r", "LASSO-r CV", "MIQP-REFIT"]
cells = Dict{Tuple{Int,String},String}()

lasso_almost = Int[]
lasso_empty  = Int[]
miqp_notopt  = Int[]

println("SparseMaxSR example — $(timestamp())")
println("Experiment A: T=$T, N=$N; methods=$(join(methods, ", "))")
R = simulate_returns(T, N)
μ, Σ = means_and_cov(R)

# Exhaustive guard: skip if too many combinations
EXH_CAP = 3_000_000
agrid = collect(0.15:0.1:0.95)

for k in ks
    # Exhaustive (guarded)
    #if binomial(N, k) ≤ EXH_CAP
    #    try
    #        _, sr, t = run_exhaustive(μ, Σ, k)
    #        cells[(k, "EXHAUSTIVE")] = cell(sr, t)
    #    catch err
    #        cells[(k, "EXHAUSTIVE")] = "ERR"
    #    end
    #else
    #    cells[(k, "EXHAUSTIVE")] = "SKIP"
    #end

    # LASSO-REFIT
    try
        _, _, sr, st, α, t = run_lasso_refit(R, μ, Σ, k; alpha=0.98)
        label = (st == :LASSO_ALLEMPTY) ? @sprintf("EMPTY / %.2fs", t) : @sprintf("%s | α*=%.2f", cell(sr, t), α)
        cells[(k,"LASSO-r")] = label
        if st == :LASSO_PATH_ALMOST_K
            push!(lasso_almost, k)
        elseif st == :LASSO_ALLEMPTY
            push!(lasso_empty, k)
        end
    catch err
        cells[(k,"LASSO-r")] = "ERR"
    end

    # LASSO-REFIT CV
    try
        _, _, sr, st, αa, t = run_lasso_refit_grid(R, μ, Σ, k; agrid=agrid)
        label = @sprintf("%s | α*=%.2f", cell(sr, t), αa)
        cells[(k,"LASSO-r CV")] = label
        if st == :LASSO_PATH_ALMOST_K
            push!(lasso_almost, k)
        end
    catch err
        cells[(k,"LASSO-r CV")] = "ERR"
    end

    # MIQP-REFIT
    try
        _, _, sr, st, t = run_miqp_refit(μ, Σ, k)
        cells[(k,"MIQP-REFIT")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt, k)
        end
    catch err
        cells[(k,"MIQP-REFIT")] = "ERR"
    end
end

print_table("Results — Experiment A (T=500, N=30)", ks, methods, cells)
println("LASSO: support size < k for k ∈ {" * _fmt_ks(lasso_almost) * "}")
println("LASSO-VANILLA: ALLEMPTY for k ∈ {" * _fmt_ks(lasso_empty) * "}")
println("MIQP: solver not OPTIMAL for k ∈ {" * _fmt_ks(miqp_notopt) * "}")
