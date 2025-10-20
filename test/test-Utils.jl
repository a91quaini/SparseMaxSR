# test/test-Utils.jl
using Test
using LinearAlgebra
using Random

using SparseMaxSR
const U = SparseMaxSR.Utils  # shorthand

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Build the explicit stabilized/symmetrized target the same way Utils._prep_S does.
function _explicit_prep_S(Σ::AbstractMatrix{<:Real}, ε::Real, stabilize::Bool)
    M = Matrix{Float64}(Σ)
    A = (M .+ M') ./ 2
    n = size(A, 1)
    if stabilize && ε > 0
        ss = ε * (tr(A) / n)
        return Symmetric(Matrix(A) .+ ss .* I(n))
    else
        return Symmetric(Matrix(A))
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "Utils.EPS_RIDGE constant" begin
    @test U.EPS_RIDGE == 1e-6
end

@testset "Utils._prep_S symmetry & stabilization" begin
    Random.seed!(123)

    # Create a generic non-symmetric matrix with positive diagonal
    A = randn(7,7)
    Σ = A * A' + 0.1I # SPD-ish starting point
    Σns = Σ .+ transpose(randn(7,7)) # break symmetry a bit

    # Case 1: stabilize=true, epsilon>0  → add ridge based on mean(diag)
    ε = 1e-3
    S1 = U._prep_S(Σns, ε, true)
    T1 = _explicit_prep_S(Σns, ε, true)

    @test typeof(S1) <: Symmetric
    @test issymmetric(Matrix(S1))
    @test isapprox(Matrix(S1), Matrix(T1); rtol=1e-12, atol=1e-12)

    # Case 2: stabilize=false (ε ignored) → just symmetrize, no ridge
    S2 = U._prep_S(Σns, ε, false)
    T2 = _explicit_prep_S(Σns, ε, false)
    @test typeof(S2) <: Symmetric
    @test issymmetric(Matrix(S2))
    @test isapprox(Matrix(S2), Matrix(T2); rtol=1e-12, atol=1e-12)

    # Case 3: epsilon=0 with stabilize=true → also no ridge, only symmetrize
    S3 = U._prep_S(Σns, 0.0, true)
    T3 = _explicit_prep_S(Σns, 0.0, true)
    @test isapprox(Matrix(S3), Matrix(T3); rtol=1e-12, atol=1e-12)
end

# ──────────────────────────────────────────────────────────────────────────────
# normalize_weights tests (replacing make_weights_sum1)
# ──────────────────────────────────────────────────────────────────────────────

@testset "Utils.normalize_weights :absolute basic scaling" begin
    # sum = 0.4 → denom = 0.4 → abs(sum(w_norm)) ≈ 1
    w = [0.2, -0.1, 0.3]
    w1 = U.normalize_weights(w; mode=:absolute, do_checks=true)
    @test isapprox(sum(w1), 1.0; atol=1e-12)
    @test isapprox.(w1, w .* (1/0.4)) |> all  # scaling factor 2.5
end

@testset "Utils.normalize_weights :relative behaves like :absolute when |s| dominates tol*||w||₁" begin
    w = [0.2, -0.1, 0.3]  # |s|=0.4, ||w||₁=0.6 → tol*||w||₁=6e-7 << 0.4
    wr = U.normalize_weights(w; mode=:relative, do_checks=true)
    @test isapprox(sum(wr), 1.0; atol=1e-12)
    @test isapprox.(wr, w .* (1/0.4)) |> all
end

@testset "Utils.normalize_weights sign preservation and negative sums (:absolute)" begin
    wneg = [-1.0, -1.0]  # sum = -2 → denom=2 → sum(w_norm) ≈ -1
    w2 = U.normalize_weights(wneg; mode=:absolute, do_checks=true)
    @test isapprox(abs(sum(w2)), 1.0; atol=1e-12)
    @test sum(w2) < 0  # sign preserved
end

@testset "Utils.normalize_weights near-zero sums produce finite outputs" begin
    # sum ≈ 0 → denom = max(|s|, tol*||w||₁, 1e-10) = 1e-10 (floor)
    wsmall = [1e-10, -1e-10]
    w0 = U.normalize_weights(wsmall; mode=:relative)  # default tol=1e-6
    @test all(isfinite, w0)
    # Check that scaling matches definition
    denom = max(abs(sum(wsmall)), 1e-6 * norm(wsmall, 1), 1e-10)
    @test isapprox.(w0, wsmall ./ denom) |> all
end

@testset "Utils.normalize_weights input validation with do_checks=true" begin
    w = [0.2, 0.3, 0.5]

    # Invalid mode
    @test_throws ErrorException U.normalize_weights(w; mode=:foo, do_checks=true)

    # Non-finite weights
    wnf = [0.1, NaN, 0.2]
    @test_throws ErrorException U.normalize_weights(wnf; do_checks=true)

    # Non-finite tol
    @test_throws ErrorException U.normalize_weights(w; tol=Inf, do_checks=true)

    # Non-positive tol
    @test_throws ErrorException U.normalize_weights(w; tol=0.0, do_checks=true)
end

@testset "Utils.normalize_weights idempotence when sum(w)=1 (:absolute)" begin
    # If sum(w)=1 and :absolute, denominator=1 → output equals input
    w = [0.25, 0.25, 0.5]   # sum = 1
    w1 = U.normalize_weights(w; mode=:absolute, do_checks=true)
    @test isapprox(sum(w1), 1.0; atol=1e-12)
    @test isapprox.(w1, w; atol=1e-12) |> all
end
