#!/usr/bin/env julia

using LinearAlgebra
using Random, Statistics
using Dates

using SparseMaxSR
using SparseMaxSR.LassoRelaxationSearch: mve_lasso_relaxation_search
using SparseMaxSR.MIQPHeuristicSearch: mve_miqp_heuristic_search

# ─────────────────────────────────────────────────────────────────────────────
# Pretty banners
# ─────────────────────────────────────────────────────────────────────────────
function banner(msg)
    println("\n", "="^90)
    println(msg)
    println("="^90, "\n")
    flush(stdout)
end

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic problems (larger to keep CPU busy long enough)
# ─────────────────────────────────────────────────────────────────────────────
function synth_returns(T::Int, N::Int; nf::Int=8, σe::Float64=0.7)
    F = randn(T, nf)
    B = 0.6 .* randn(N, nf)
    E = σe .* randn(T, N)
    return F * B' .+ E
end

function synth_moments(N::Int; shiftμ=0.015, scaleμ=0.04, jitterΣ=0.08)
    μ = shiftμ .+ scaleμ .* randn(N)
    A = randn(N, N)
    Σ = Symmetric(A * A' + jitterΣ * I(N)) # NOTE: no broadcasting with I
    return μ, Σ
end

# ─────────────────────────────────────────────────────────────────────────────
# CPU monitoring helpers (Linux-only, no extra packages)
#   • system_cpu_util(interval): avg % utilization across cores for 'interval'
#   • proc_cpu_and_wall(f): run f(); returns (cpu_seconds, wall_seconds)
# ─────────────────────────────────────────────────────────────────────────────
function _parse_cpu_line(line)
    nums = parse.(UInt128, split(line)[2:end]) # user,nice,system,idle,iowait,irq,softirq,steal,guest,guest_nice
    total = sum(nums)
    idle  = nums[4] + (length(nums) >= 5 ? nums[5] : 0) # idle+iowait
    return total, idle
end

function system_cpu_util(; interval::Real=5.0)
    lines1 = readlines("/proc/stat")
    sleep(interval)
    lines2 = readlines("/proc/stat")

    # first line is 'cpu ' aggregate; next are per-core 'cpu0','cpu1',...
    ncores = Sys.CPU_THREADS
    take1 = lines1[1:(min(ncores+1, length(lines1)))]
    take2 = lines2[1:(min(ncores+1, length(lines2)))]

    util = Float64[]
    for i in 2:length(take1) # per-core only (skip aggregate)
        t1,i1 = _parse_cpu_line(take1[i])
        t2,i2 = _parse_cpu_line(take2[i])
        push!(util, 100 * (1 - Float64(i2 - i1) / Float64(t2 - t1)))
    end
    return mean(util), util
end

# process CPU time in seconds (user+system) using /proc/self/stat
const _CLK_TCK = try
    Base.Libc.sysconf(Base.Libc._SC_CLK_TCK)
catch
    100 # fallback, typical but not guaranteed
end

function _proc_cpu_jiffies()
    fields = split(read("/proc/self/stat", String))
    utime  = parse(UInt128, fields[14])
    stime  = parse(UInt128, fields[15])
    return utime + stime
end

function proc_cpu_and_wall(f::Function)
    u1 = _proc_cpu_jiffies()
    t1 = time()
    f()
    t2 = time()
    u2 = _proc_cpu_jiffies()
    return (Float64(u2 - u1) / _CLK_TCK, t2 - t1)
end

# Wrap a workload 'body()' with CPU monitoring and print results
function run_with_monitor(title::String, body::Function; sys_interval=6.0)
    banner(title)
    println("Julia Threads.nthreads() = ", Threads.nthreads())
    println("BLAS.get_num_threads()   = ", try BLAS.get_num_threads() catch; "unknown" end)
    println("Sys.CPU_THREADS          = ", Sys.CPU_THREADS)
    println("Monitoring interval      = ", sys_interval, " seconds")
    flush(stdout)

    # start system monitor concurrently
    sys_task = @async system_cpu_util(interval=sys_interval)

    # run workload and measure process CPU vs wall
    cpu_s, wall_s = proc_cpu_and_wall(body)

    # wait for system monitor
    avg_util, per_core = fetch(sys_task)

    println("\nResults:")
    println("  • Process CPU time (user+system): $(round(cpu_s, digits=3)) s")
    println("  • Wall time:                      $(round(wall_s, digits=3)) s")
    if wall_s > 0
        println("  • CPU/Wall ratio:                 ", round(cpu_s / wall_s, digits=2),
                "  (≈ #cores concurrently used by this process)")
    end
    println("  • System avg per-core util:       $(round(avg_util, digits=1))% over $(sys_interval)s")
    println("  • Per-core (%):                   ",
            join(map(x->string(round(x,digits=0)), per_core), ", "))
    flush(stdout)
end

# place right below your existing run_with_monitor(::String, ::Function; ...) method
run_with_monitor(body::Function, title::String; sys_interval=6.0) =
    run_with_monitor(title, body; sys_interval=sys_interval)

# ─────────────────────────────────────────────────────────────────────────────
# Heavier LASSO/MIQP workloads (repeat loops to ensure ≥ several seconds)
# ─────────────────────────────────────────────────────────────────────────────
function lasso_heavy(R; k::Int, alpha, refit::Bool, reps::Int)
    for r in 1:reps
        _sel, _w, _sr, _status, _α = mve_lasso_relaxation_search(
            R; k=k, alpha=alpha, compute_weights=true,
            use_refit=refit, cv_folds=5, standardize=false, do_checks=true
        )
        if r % max(1, reps ÷ 10) == 0
            print("."); flush(stdout)
        end
    end
    println()
end

function miqp_heavy(μ, Σ; k::Int, threads::Int, reps::Int)
    for r in 1:reps
        _res = mve_miqp_heuristic_search(
            μ, Σ; k=k, γ=1.0, stabilize_Σ=false,
            compute_weights=true, use_refit=false,
            threads=threads, exactly_k=true, normalize_weights=true
        )
        if r % max(1, reps ÷ 10) == 0
            print("."); flush(stdout)
        end
    end
    println()
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
function main()
    Random.seed!(20251104)

    # Bigger LASSO problem to keep BLAS busy longer
    T, N_lasso = 5000, 1200           # sizeable (≈ a few seconds per section)
    R = synth_returns(T, N_lasso; nf=10, σe=0.7)
    k_lasso     = 25
    # Alpha is elastic-net mixing; workload comes from data size & reps
    alpha_grid  = [0.2, 0.5, 0.8, 1.0]
    lasso_reps  = 8

    # Bigger MIQP (still moderate so it finishes reasonably fast)
    N_miqp, k_miqp = 60, 10
    μ, Σ = synth_moments(N_miqp; shiftμ=0.01, scaleμ=0.03, jitterΣ=0.06)
    miqp_reps = 6

    # Section A: LASSO with DEFAULT BLAS
    run_with_monitor("Section A: LASSO (BLAS default)") do
        lasso_heavy(R; k=k_lasso, alpha=alpha_grid, refit=true, reps=lasso_reps)
    end

    # Section B: LASSO with BLAS=1
    try
        BLAS.set_num_threads(1)
        println("\nBLAS threads forced to: ", BLAS.get_num_threads())
    catch
        println("\nThis BLAS does not support set_num_threads(); continuing anyway.")
    end
    run_with_monitor("Section B: LASSO (BLAS=1)") do
        lasso_heavy(R; k=k_lasso, alpha=alpha_grid, refit=true, reps=lasso_reps)
    end

    # Section C: MIQP with threads=1
    run_with_monitor("Section C: MIQP (threads=1)") do
        miqp_heavy(μ, Σ; k=k_miqp, threads=1, reps=miqp_reps)
    end

    # Section D: MIQP with threads=3
    run_with_monitor("Section D: MIQP (threads=3)") do
        miqp_heavy(μ, Σ; k=k_miqp, threads=3, reps=miqp_reps)
    end

    # Section E: MIQP with threads=0 (auto)
    run_with_monitor("Section E: MIQP (threads=0 = auto)") do
        miqp_heavy(μ, Σ; k=k_miqp, threads=0, reps=miqp_reps)
    end

    banner("Done.")
end

main()
