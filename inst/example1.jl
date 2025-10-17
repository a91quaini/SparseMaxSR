#!/usr/bin/env julia
# Extended example: compare Exhaustive, LASSO (vanilla/refit), and MIQP (vanilla/refit)
# - Experiment A:  T=500, N=30,   k ∈ {1,3,5,7,9}       (Exhaustive + LASSO + MIQP)
# - Experiment B:  T=500, N=100,  k ∈ {1,5,10,15,20,…}  (LASSO + MIQP)
# - Experiment C:  T=1250, N=500, k ∈ {100, 500}        (LASSO + MIQP)
# - Experiment D:  T=500,  N=1000, k ∈ {500}            (LASSO + MIQP)
#
# Run:  julia --project=. example1.jl
#
# Notes:
# * Exhaustive search is skipped automatically when C(N,k) exceeds a cap.
# * Cells show "SR / time_s", where SR is the Sharpe ratio and time_s is seconds.
# * LASSO-VANILLA is the normalize-coeffs branch (abs(sum(w))=1 if possible; else :LASSO_ALLEMPTY).
# * LASSO-REFIT refits exact MVE on the selected support.
# * MIQP-VANILLA reports the MIQP solution as returned by the heuristic.
# * MIQP-REFIT refits exact MVE on the MIQP-selected support (if the routine supports it).

using SparseMaxSR
using Random, LinearAlgebra, Statistics, Printf, Dates
using Combinatorics: binomial
import MathOptInterface as MOI  # for checking MIQP optimality symbol

# -----------------------
# Helpers
# -----------------------

timestamp() = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")

# Simulate returns with a mild 2-factor structure + noise
function simulate_returns(T::Int, N::Int; nf::Int=2, beta_scale=0.3, eps_scale=0.7, rng = Random.default_rng())
    F = randn(rng, T, nf)
    B = beta_scale .* randn(rng, N, nf)
    E = eps_scale  .* randn(rng, T, N)
    R = F * B' .+ E
    return R
end

# Moments from returns
means_and_cov(R) = (vec(mean(R, dims=1)), cov(R; corrected=true))

# Compact formatter: "SR / time_s"
cell(sr, t) = isnan(sr) ? "-" : @sprintf("%.4f / %.2fs", sr, t)

# Pretty list for trackers
_fmt_ks(v::Vector{Int}) = isempty(v) ? "(none)" : join(sort(unique(v)), ", ")

# -----------------------
# Method runners
# -----------------------

# Exhaustive: (μ, Σ, k; exactly_k=true)
function run_exhaustive(μ, Σ, k)
    sel = w = nothing; sr = NaN; st = :UNKNOWN
    tsec = @elapsed begin
        sel, w, sr, st = SparseMaxSR.mve_exhaustive_search(μ, Σ, k; exactly_k=true)
    end
    return sel, w, sr, tsec, st
end

# MIQP heuristic — VANILLA (no refit), returns what the MIQP heuristic solved
function run_miqp_vanilla(μ, Σ, k)
    sel = w = nothing; sr = NaN; st = :UNKNOWN
    tsec = @elapsed begin
        sel, w, sr, st = SparseMaxSR.mve_miqp_heuristic_search(
            μ, Σ; k=k,
            compute_weights=true,
            use_refit=false
        )
    end
    return sel, w, sr, tsec, st
end

# MIQP heuristic — REFIT (refit exact MVE on MIQP support)
function run_miqp_refit(μ, Σ, k)
    sel = w = nothing; sr = NaN; st = :UNKNOWN
    tsec = @elapsed begin
        sel, w, sr, st = SparseMaxSR.mve_miqp_heuristic_search(
            μ, Σ; k=k,
            compute_weights=true,
            use_refit=true
        )
    end
    return sel, w, sr, tsec, st
end

# LASSO-VANILLA (normalize-coeffs, use_refit=false). Uses moment-only entry (μ,Σ,T).
function run_lasso_vanilla(R, μ, Σ, k; alpha=0.95)
    sel = w = nothing; sr = NaN; st = :UNKNOWN
    tsec = @elapsed begin
        sel, w, sr, st = SparseMaxSR.mve_lasso_relaxation_search(
            μ, Σ, size(R,1);
            # the following kwargs reflect the "path then pick" design; keep tuned as per your current API
            k = k,
            nlambda = 100,
            lambda_min_ratio = 1e-5,
            alpha = alpha,
            standardize = false,
            epsilon = SparseMaxSR.EPS_RIDGE[],   # dereference Ref
            stabilize_Σ = true,
            compute_weights = true,     # ignored in vanilla branch; harmless
            use_refit = false,          # vanilla = normalized β
            do_checks = false
        )
    end
    return sel, w, sr, tsec, st
end

# LASSO-REFIT (use_refit=true, compute_weights=true). Uses moment-only entry (μ,Σ,T).
function run_lasso_refit(R, μ, Σ, k; alpha=0.95)
    sel = w = nothing; sr = NaN; st = :UNKNOWN
    tsec = @elapsed begin
        sel, w, sr, st = SparseMaxSR.mve_lasso_relaxation_search(
            μ, Σ, size(R,1);
            k = k,
            nlambda = 100,
            lambda_min_ratio = 1e-5,
            alpha = alpha,
            standardize = false,
            epsilon = SparseMaxSR.EPS_RIDGE[],   # dereference Ref
            stabilize_Σ = true,
            compute_weights = true,     # request refit weights
            use_refit = true,           # refit MVE on support
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

# =============================================================================
# Experiment A: T=500, N=30, k in 1,3,5,7,9
# =============================================================================
Random.seed!(42)
T, N = 500, 30
ks = [1,3,5,7,9]
methods = ["EXHAUSTIVE", "LASSO-VANILLA", "LASSO-REFIT", "MIQP-VANILLA", "MIQP-REFIT"]
cells = Dict{Tuple{Int,String},String}()

# trackers
lasso_almost_A  = Int[]
lasso_empty_A   = Int[]
miqp_notopt_A   = Int[]

println("SparseMaxSR example — $(timestamp())")
println("Experiment A: T=$T, N=$N; methods=$(join(methods, ", "))")
R = simulate_returns(T, N)
μ, Σ = means_and_cov(R)

# Exhaustive guard: skip if too many combinations
EXH_CAP = 3_000_000

for k in ks
    # Exhaustive (guarded)
    if binomial(N,k) <= EXH_CAP
        try
            _, _, sr, t, _ = run_exhaustive(μ, Σ, k)
            cells[(k,"EXHAUSTIVE")] = cell(sr, t)
        catch
            cells[(k,"EXHAUSTIVE")] = "ERR"
        end
    else
        cells[(k,"EXHAUSTIVE")] = "SKIP"
    end

    # LASSO-VANILLA
    try
        _, _, sr, t, st = run_lasso_vanilla(R, μ, Σ, k; alpha=0.99)
        cells[(k,"LASSO-VANILLA")] = (st == :LASSO_ALLEMPTY) ? @sprintf("%-18s", "EMPTY / $(round(t; digits=2))s") : cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K
            push!(lasso_almost_A, k)
        elseif st == :LASSO_ALLEMPTY
            push!(lasso_empty_A, k)
        end
    catch
        cells[(k,"LASSO-VANILLA")] = "ERR"
    end

    # LASSO-REFIT
    try
        _, _, sr, t, st = run_lasso_refit(R, μ, Σ, k; alpha=0.99)
        cells[(k,"LASSO-REFIT")] = cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K
            push!(lasso_almost_A, k)
        end
    catch
        cells[(k,"LASSO-REFIT")] = "ERR"
    end

    # MIQP-VANILLA
    try
        _, _, sr, t, st = run_miqp_vanilla(μ, Σ, k)
        cells[(k,"MIQP-VANILLA")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_D, k)
        end
    catch
        cells[(k,"MIQP-VANILLA")] = "ERR"
    end

    # MIQP-REFIT
    try
        _, _, sr, t, st = run_miqp_refit(μ, Σ, k)
        cells[(k,"MIQP-REFIT")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_D, k)
        end
    catch
        cells[(k,"MIQP-REFIT")] = "ERR"
    end

end

print_table("Results — Experiment A (T=500, N=30)", ks, methods, cells)
println("LASSO (both): support size < k for k ∈ {" * _fmt_ks(lasso_almost_A) * "}")
println("LASSO-VANILLA: ALLEMPTY for k ∈ {" * _fmt_ks(lasso_empty_A) * "}")
println("MIQP: solver not OPTIMAL for k ∈ {" * _fmt_ks(miqp_notopt_A) * "}")

# =============================================================================
# Experiment B: T=500, N=100, k in 1,5,10,15,20,…
# =============================================================================
Random.seed!(1729)
T, N = 500, 100
ks = [1; collect(5:5:70)]
methods = ["LASSO-VANILLA", "LASSO-REFIT", "MIQP-VANILLA", "MIQP-REFIT"]
cells = Dict{Tuple{Int,String},String}()

lasso_almost_B = Int[]
lasso_empty_B  = Int[]
miqp_notopt_B  = Int[]

println("Experiment B: T=$T, N=$N; methods=$(join(methods, ", "))")
R = simulate_returns(T, N)
μ, Σ = means_and_cov(R)

for k in ks
    # LASSO-VANILLA
    try
        _, _, sr, t, st = run_lasso_vanilla(R, μ, Σ, k; alpha=0.95)
        cells[(k,"LASSO-VANILLA")] = (st == :LASSO_ALLEMPTY) ? @sprintf("%-18s", "EMPTY / $(round(t; digits=2))s") : cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K
            push!(lasso_almost_B, k)
        elseif st == :LASSO_ALLEMPTY
            push!(lasso_empty_B, k)
        end
    catch
        cells[(k,"LASSO-VANILLA")] = "ERR"
    end

    # LASSO-REFIT
    try
        _, _, sr, t, st = run_lasso_refit(R, μ, Σ, k; alpha=0.95)
        cells[(k,"LASSO-REFIT")] = cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K
            push!(lasso_almost_B, k)
        end
    catch
        cells[(k,"LASSO-REFIT")] = "ERR"
    end

    # MIQP-VANILLA
    try
        _, _, sr, t, st = run_miqp_vanilla(μ, Σ, k)
        cells[(k,"MIQP-VANILLA")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_B, k)
        end
    catch
        cells[(k,"MIQP-VANILLA")] = "ERR"
    end

    # MIQP-REFIT
    try
        _, _, sr, t, st = run_miqp_refit(μ, Σ, k)
        cells[(k,"MIQP-REFIT")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_B, k)
        end
    catch
        cells[(k,"MIQP-REFIT")] = "ERR"
    end
end

print_table("Results — Experiment B (T=500, N=100)", ks, methods, cells)
println("LASSO (both): support size < k for k ∈ {" * _fmt_ks(lasso_almost_B) * "}")
println("LASSO-VANILLA: ALLEMPTY for k ∈ {" * _fmt_ks(lasso_empty_B) * "}")
println("MIQP: solver not OPTIMAL for k ∈ {" * _fmt_ks(miqp_notopt_B) * "}")

# =============================================================================
# Experiment C: T=250*5 (=1250), N=500, k in {100, 500}
# =============================================================================
Random.seed!(2025)
T, N = 250*5, 500
ks = [100, 500]
methods = ["LASSO-VANILLA", "LASSO-REFIT", "MIQP-VANILLA", "MIQP-REFIT"]
cells = Dict{Tuple{Int,String},String}()

lasso_almost_C = Int[]
lasso_empty_C  = Int[]
miqp_notopt_C  = Int[]

println("Experiment C: T=$T, N=$N; methods=$(join(methods, ", "))")
R = simulate_returns(T, N)
μ, Σ = means_and_cov(R)

for k in ks
    # LASSO-VANILLA
    try
        _, _, sr, t, st = run_lasso_vanilla(R, μ, Σ, k; alpha=0.95)
        cells[(k,"LASSO-VANILLA")] = (st == :LASSO_ALLEMPTY) ? @sprintf("%-18s", "EMPTY / $(round(t; digits=2))s") : cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K
            push!(lasso_almost_C, k)
        elseif st == :LASSO_ALLEMPTY
            push!(lasso_empty_C, k)
        end
    catch
        cells[(k,"LASSO-VANILLA")] = "ERR"
    end

    # LASSO-REFIT
    try
        _, _, sr, t, st = run_lasso_refit(R, μ, Σ, k; alpha=0.95)
        cells[(k,"LASSO-REFIT")] = cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K
            push!(lasso_almost_C, k)
        end
    catch
        cells[(k,"LASSO-REFIT")] = "ERR"
    end

    # MIQP-VANILLA
    try
        _, _, sr, t, st = run_miqp_vanilla(μ, Σ, k)
        cells[(k,"MIQP-VANILLA")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_C, k)
        end
    catch
        cells[(k,"MIQP-VANILLA")] = "ERR"
    end

    # MIQP-REFIT
    try
        _, _, sr, t, st = run_miqp_refit(μ, Σ, k)
        cells[(k,"MIQP-REFIT")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_C, k)
        end
    catch
        cells[(k,"MIQP-REFIT")] = "ERR"
    end
end

print_table("Results — Experiment C (T=1250, N=500)", ks, methods, cells)
println("LASSO (both): support size < k for k ∈ {" * _fmt_ks(lasso_almost_C) * "}")
println("LASSO-VANILLA: ALLEMPTY for k ∈ {" * _fmt_ks(lasso_empty_C) * "}")
println("MIQP: solver not OPTIMAL for k ∈ {" * _fmt_ks(miqp_notopt_C) * "}")

# =============================================================================
# Experiment D: T=250*2 (=500), N=1000, k in {500}
# =============================================================================
Random.seed!(2025)
T, N = 250*2, 1000
ks = [500]
methods = ["LASSO-VANILLA", "LASSO-REFIT", "MIQP-VANILLA", "MIQP-REFIT"]
cells = Dict{Tuple{Int,String},String}()

lasso_almost_D = Int[]
lasso_empty_D  = Int[]
miqp_notopt_D  = Int[]

println("Experiment D: T=$T, N=$N; methods=$(join(methods, ", "))")
R = simulate_returns(T, N)
μ, Σ = means_and_cov(R)

for k in ks
    # LASSO-VANILLA
    try
        _, _, sr, t, st = run_lasso_vanilla(R, μ, Σ, k; alpha=0.6)
        cells[(k,"LASSO-VANILLA")] = (st == :LASSO_ALLEMPTY) ? @sprintf("%-18s", "EMPTY / $(round(t; digits=2))s") : cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K
            push!(lasso_almost_D, k)
        elseif st == :LASSO_ALLEMPTY
            push!(lasso_empty_D, k)
        end
    catch
        cells[(k,"LASSO-VANILLA")] = "ERR"
    end

    # LASSO-REFIT
    try
        _, _, sr, t, st = run_lasso_refit(R, μ, Σ, k; alpha=0.6)
        cells[(k,"LASSO-REFIT")] = cell(sr, t)
        if st == :LASSO_PATH_ALMOST_K
            push!(lasso_almost_D, k)
        end
    catch
        cells[(k,"LASSO-REFIT")] = "ERR"
    end

    # MIQP-VANILLA
    try
        _, _, sr, t, st = run_miqp_vanilla(μ, Σ, k)
        cells[(k,"MIQP-VANILLA")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_D, k)
        end
    catch
        cells[(k,"MIQP-VANILLA")] = "ERR"
    end

    # MIQP-REFIT
    try
        _, _, sr, t, st = run_miqp_refit(μ, Σ, k)
        cells[(k,"MIQP-REFIT")] = cell(sr, t)
        if st != MOI.OPTIMAL
            push!(miqp_notopt_D, k)
        end
    catch
        cells[(k,"MIQP-REFIT")] = "ERR"
    end
end

print_table("Results — Experiment D (T=500, N=1000)", ks, methods, cells)
println("LASSO (both): support size < k for k ∈ {" * _fmt_ks(lasso_almost_D) * "}")
println("LASSO-VANILLA: ALLEMPTY for k ∈ {" * _fmt_ks(lasso_empty_D) * "}")
println("MIQP: solver not OPTIMAL for k ∈ {" * _fmt_ks(miqp_notopt_D) * "}")
