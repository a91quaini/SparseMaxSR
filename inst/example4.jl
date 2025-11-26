#!/usr/bin/env julia

using Random, LinearAlgebra, Statistics, Printf, Dates
using SparseMaxSR

# ---------------------------
# Config (globals are fine)
# ---------------------------
const SEED  = 123
const T     = 200
const N     = 50
const W_IN  = 40
const W_OUT = 20
const STEP  = 20
const K_GRID = [5, 10, 15, 20]

dfmax_for(k) = k + 5
pmax_for(k)  = 2k

const EPS = 1e-8
const NORMALIZE_WEIGHTS = false

# ---------------------------
# Helpers (pure functions)
# ---------------------------
sharpe(x) = (σ = std(x); σ > 0 ? mean(x) / σ : 0.0)
nnz_tol(x; tol=1e-10) = count(abs.(x) .> tol)
maybe_normalize(x; normalize::Bool) = normalize ? (abs(sum(x)) > EPS ? x ./ sum(x) : x) : x

function simulate_returns(T, N; seed=SEED)
    Random.seed!(seed)
    Λ = 0.6 .* randn(N, 3)
    F = randn(T, 3)
    ε = 0.4 .* randn(T, N)
    R = F * Λ' .+ ε
    return R
end

function insample_moments(R_in)
    μ = vec(mean(R_in, dims=1))
    Σ = cov(R_in; corrected=true)
    return μ, Symmetric((Σ + Σ') / 2)
end

function lasso_weights(μ, Σ; k, T_in::Int=W_IN, normalize::Bool)
    # Use moment-based API: (μ, Σ, T) — vanilla alpha=1, fixed
    res = SparseMaxSR.mve_lasso_relaxation_search(
        μ, Σ, T_in;
        k = k,
        R = nothing,              # not doing α-CV on OOS design here
        nlambda = 300,
        lambda_min_ratio = 1e-2,
        lambda = nothing,
        alpha = 1.0,
        alpha_select = :fixed,    # vanilla LASSO (α=1) as requested
        nadd = 80,
        nnested = 3,
        standardize = false,
        epsilon = 1e-8,
        stabilize_Σ = true,
        compute_weights = true,
        normalize_weights = normalize,
        use_refit = false,
        do_checks = true,
        cv_folds = 5,
        cv_verbose = false,
        gcv_kappa = 1.0
    )

    # API returns (selection, weights, sr, status, alpha)
    w = res[:weights]
    return (weights = w, nnz = count(abs.(w) .> 1e-10))
end


function miqp_weights(μ, Σ; k, normalize)
    r = SparseMaxSR.mve_miqp_heuristic_search(
        μ, Σ;
        k=k, exactly_k=false, γ=1.0,
        stabilize_Σ=true, epsilon=1e-8,
        fmin=fill(-Inf, length(μ)), fmax=fill(Inf, length(μ)),
        compute_weights=true, use_refit=false, threads=0,
        normalize_weights=normalize, do_checks=true, verbose=false
    )
    w = r[:weights]
    return (weights=w, nnz=nnz_tol(w))
end

# ---------------------------
# Runnable block (no soft-scope issues)
# ---------------------------
println("Started at ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))

let
    # simulate once
    R = simulate_returns(T, N)

    # build windows using local names
    windows = Tuple{UnitRange{Int},UnitRange{Int}}[]
    tstart = 1
    while true
        is_end  = tstart + W_IN - 1
        oos_end = is_end + W_OUT
        oos_end <= T || break
        push!(windows, (tstart:is_end, (is_end+1):oos_end))
        tstart += STEP
    end
    println("Rolling windows: ", length(windows))

    # collectors
    oos_rets = Dict{Tuple{Symbol,Int}, Vector{Float64}}()
    runtimes = Dict{Tuple{Symbol,Int}, Float64}()
    card_log = Dict{Tuple{Symbol,Int,Int}, Int}()

    for k in K_GRID
        oos_rets[(:lasso,k)] = Float64[]
        oos_rets[(:miqp, k)] = Float64[]
        runtimes[(:lasso,k)] = 0.0
        runtimes[(:miqp, k)] = 0.0
    end

    # roll
    for (w_idx, (idx_in, idx_out)) in enumerate(windows)
        μ, Σ = insample_moments(@view R[idx_in,:])
        R_out = @view R[idx_out,:]
        for k in K_GRID
            t₀ = time(); l = lasso_weights(μ, Σ; k=k, normalize=NORMALIZE_WEIGHTS); t₁ = time()
            runtimes[(:lasso,k)] += t₁ - t₀
            card_log[(:lasso,k,w_idx)] = l.nnz
            append!(oos_rets[(:lasso,k)], Vector(R_out * l.weights))

            t₀ = time(); m = miqp_weights(μ, Σ; k=k, normalize=NORMALIZE_WEIGHTS); t₁ = time()
            runtimes[(:miqp,k)] += t₁ - t₀
            card_log[(:miqp,k,w_idx)] = m.nnz
            append!(oos_rets[(:miqp,k)], Vector(R_out * m.weights))
        end
    end

    # report 1: OOS SR and time
    println("\nOOS Sharpe Ratios and total time:")
    println(rpad("Method",10), rpad("k",5), rpad("Sharpe",10), "Time(s)")
    println("-"^35)
    for k in K_GRID
        for m in [:lasso, :miqp]
            sr = sharpe(oos_rets[(m,k)])
            println(rpad(String(m),10), rpad(string(k),5),
                    @sprintf("%8.4f", sr), "   ",
                    @sprintf("%8.2f", runtimes[(m,k)]))
        end
    end

    # report 2: realized cardinality by window
    println("\nRealized cardinality by window:")
    println(rpad("Method",10), rpad("k",5), rpad("Window",8), "nnz")
    println("-"^35)
    for k in K_GRID, w in 1:length(windows), m in [:lasso, :miqp]
        nnz = card_log[(m,k,w)]
        println(rpad(String(m),10), rpad(string(k),5),
                rpad(string(w),8), nnz)
    end
end
