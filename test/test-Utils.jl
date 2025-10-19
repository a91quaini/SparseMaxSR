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

@testset "Utils.make_weights_sum1 basic scaling (:sum)" begin
    w = [0.2, -0.1, 0.3] # sum = 0.4
    w1, status, s = U.make_weights_sum1(w; target=1.0, mode=:sum, do_checks=true)
    @test status == :OK
    @test isapprox(s, 0.4; atol=1e-15)
    @test isapprox(sum(w1), 1.0; atol=1e-12)
    # scaling factor should be 2.5
    @test isapprox.(w1, w .* 2.5) |> all
end

@testset "Utils.make_weights_sum1 absolute-sum normalization (:abs)" begin
    # Negative sum → preserve sign after scaling
    wneg = [-1.0, -1.0] # sum = -2
    w2, status2, s2 = U.make_weights_sum1(wneg; target=1.0, mode=:abs, do_checks=true)
    @test status2 == :OK
    @test isapprox(abs(sum(w2)), 1.0; atol=1e-12)
    @test sum(w2) < 0 # sign preserved

    # Positive sum → preserve positive sign
    wpos = [0.3, 0.7] # sum = 1.0
    w3, status3, s3 = U.make_weights_sum1(wpos; target=2.0, mode=:abs, do_checks=true)
    @test status3 == :OK
    @test isapprox(abs(sum(w3)), 2.0; atol=1e-12)
    @test sum(w3) > 0
end

@testset "Utils.make_weights_sum1 near-zero/degenerate sums" begin
    # Sum ~ 0 → ZERO_SUM and zeros vector
    w = [1e-10, -1e-10]
    w0, st, s = U.make_weights_sum1(w; target=1.0, mode=:sum)  # default tol=1e-7
    @test st == :ZERO_SUM
    @test sum(abs, w0) == 0.0

    # Non-finite or non-positive target without checks → ZERO_SUM branch
    w = [0.4, 0.6]
    w_bad, st_bad, s_bad = U.make_weights_sum1(w; target=NaN, mode=:sum, do_checks=false)
    @test st_bad == :ZERO_SUM
    @test sum(abs, w_bad) == 0.0

    w_bad2, st_bad2, _ = U.make_weights_sum1(w; target=-1.0, mode=:sum, do_checks=false)
    @test st_bad2 == :ZERO_SUM
    @test sum(abs, w_bad2) == 0.0
end

@testset "Utils.make_weights_sum1 input validation with do_checks=true" begin
    w = [0.2, 0.3, 0.5]

    # Invalid mode
    @test_throws ErrorException U.make_weights_sum1(w; mode=:foo, do_checks=true)

    # Non-finite weights
    wnf = [0.1, NaN, 0.2]
    @test_throws ErrorException U.make_weights_sum1(wnf; do_checks=true)

    # Non-finite target
    @test_throws ErrorException U.make_weights_sum1(w; target=Inf, do_checks=true)

    # Non-positive target
    @test_throws ErrorException U.make_weights_sum1(w; target=0.0, do_checks=true)
end

@testset "Utils.make_weights_sum1 idempotence under correct target" begin
    # If sum(w) already equals target and mode=:sum, output equals input
    w = [0.25, 0.25, 0.5]
    w1, st, s = U.make_weights_sum1(w; target=1.0, mode=:sum, do_checks=true)
    @test st == :OK
    @test isapprox(sum(w1), 1.0; atol=1e-12)
    @test isapprox.(w1, w) |> all
end
