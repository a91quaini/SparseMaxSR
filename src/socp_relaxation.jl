# returns a valid lower bound on the master obj via an SOCP relaxation
function portfolios_socp(μ, Σ, A, l, u, mininv, k::Int)
    n = length(μ)
    m = size(A,1)

    model = Model(optimizer_with_attributes(
      MosekTools.Optimizer,
      "MSK_DPAR_INTPNT_QO_TOL_PFEAS" => 1e-6,
      "MSK_DPAR_INTPNT_QO_TOL_DFEAS" => 1e-6,
      "MSK_IPAR_LOG"              => 0
    ))

    @variable(model, α[1:n])
    @variable(model, λ)
    @variable(model, v[1:n] >= 0)
    @variable(model, t >= 0)
    @variable(model, w[1:n])
    @variable(model, βl[1:m] >= 0)
    @variable(model, βu[1:m] >= 0)
    @variable(model, ρ[1:n] >= 0)

    # linking constraints w_i >= Σ_i'*α + A[:,i]'(βl-βu) + λ + ρ_i - mininv_i
    Σα = Σ * α
    for i in 1:n
      @constraint(model,
        w[i] >= Σα[i]
                   + A[:,i]'*(βl - βu)
                   + λ
                   + ρ[i]
                   - mininv[i]
      )
      # SOCP epigraph: v[i] + t + ρ[i]*mininv[i] >= ½ w[i]^2
      @constraint(model,
        v[i] + t + ρ[i]*mininv[i] >= (1/2)*w[i]^2
      )
    end

    @constraint(model, sum(ρ) <= k)  # convex envelope of sum(s)=k
    @constraint(model, sum(mininv[i]*ρ[i] for i=1:n) <= 1)

    @objective(model, Max,
      -0.5*dot(α,α)
      + dot(μ, α)
      + λ
      + dot(βl, l)
      - dot(βu, u)
      - sum(v)
      - k*t
    )

    optimize!(model)
    return objective_value(model)
end
