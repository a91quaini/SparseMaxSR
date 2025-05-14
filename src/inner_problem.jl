# solve the dual QP for a fixed support “inds”
function inner_dual(μ::Vector{Float64},
                    Σ::Matrix{Float64},
                    A::Matrix{Float64},
                    l::Vector{Float64},
                    u::Vector{Float64},
                    mininv::Vector{Float64},
                    inds::Vector{Int})
    n = length(μ)
    m = size(A,1)
    f = length(inds)

    model = Model(optimizer_with_attributes(
        MosekTools.Optimizer,
        "MSK_DPAR_INTPNT_QO_TOL_PFEAS" => 1e-8,
        "MSK_DPAR_INTPNT_QO_TOL_DFEAS" => 1e-8,
        "MSK_IPAR_LOG"              => 0
    ))

    @variable(model, α[1:n])
    @variable(model, λ)
    @variable(model, ρ[1:f] >= 0)
    @variable(model, βl[1:m] >= 0)
    @variable(model, βu[1:m] >= 0)
    @variable(model, w[1:f])

    # build Σ*α once
    Σα = Σ * α

    # cut constraint: w_j >= ... for each j in inds
    for (j,i) in enumerate(inds)
        @constraint(model,
          w[j] >= Σα[i]
                     + A[:,i]'*(βl - βu)
                     + λ
                     + ρ[j]
                     - mininv[i]
        )
    end

    @objective(model, Max,
      -0.5 * dot(α,α)
      - 0.5 * sum(w[j]^2 for j=1:f)
      + dot(μ, α)
      + λ
      + dot(βl, l)
      - dot(βu, u)
      + dot(ρ, mininv[inds])
    )

    optimize!(model)
    return (
      ofv    = objective_value(model),
      α      = value.(α),
      λ      = value(λ),
      βl     = value.(βl),
      βu     = value.(βu),
      ρ      = value.(ρ),
      w      = value.(w),
      status = termination_status(model),
    )
end
