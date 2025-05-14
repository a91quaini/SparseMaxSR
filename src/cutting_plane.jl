using MathOptInterface
const MOI = MathOptInterface

function cutting_plane_portfolio(μ, Σ, A, l, u, mininv, k;
                                 time_limit=600.0,
                                 mip_gap=1e-4,
                                 use_warmstart=true,
                                 use_socp=true)

  n = length(μ)
  model = Model(optimizer_with_attributes(
    CPLEX.Optimizer,
    "CPX_PARAM_TILIM" => time_limit,
    "CPX_PARAM_EPGAP" => mip_gap
  ))

  @variable(model, s[1:n], Bin)
  @variable(model, t)

  @objective(model, Min, t)
  @constraint(model, sum(s) <= k)
  @constraint(model, sum(s) >= 1)
  @constraint(model, sum(mininv[i]*s[i] for i=1:n) <= 1)

  # optional: get a socp lower bound
  socp_lb = use_socp ? portfolios_socp(μ,Σ,A,l,u,mininv,k) : -Inf

  # optional warm‐start: pick a good support
  if use_warmstart
    inds0 = sort(sample(1:n, k; replace=false))
    warm_inds = portfolios_hillclimb(μ,Σ,A,l,u,mininv,k,inds0)
    # set initial s
    for i in warm_inds
      set_start_value(s[i], 1.0)
    end
    # add the single cut at the start
    p0, grad0 = begin
      D = inner_dual(μ,Σ,A,l,u,mininv,warm_inds)
      # assemble gradient
      g = zeros(n)
      for (j,i) in enumerate(warm_inds)
        g[i] = -0.5*D.w[j]^2 + D.ρ[j]*mininv[i]
      end
      (D.ofv, g)
    end
    @constraint(model, t >= p0 + dot(grad0, s .- zeros(n) .- (s .- 1).*0))  # stabilized at s=warm_inds
  end

  # lazy‐cut callback
  function _lazy(cb_data)
    if callback_node_status(cb_data) != MOI.SOLVER_NODE_STATUS_ONLY_INTEGER
      return
    end
    s_val = callback_value.(cb_data, s) .> 0.5
    inds = findall(s_val)
    D = inner_dual(μ,Σ,A,l,u,mininv,inds)
    p = D.ofv
    g = zeros(n)
    for (j,i) in enumerate(inds)
      g[i] = -0.5*D.w[j]^2 + D.ρ[j]*mininv[i]
    end
    @lazy_constraint(cb_data, t >= p + dot(g, s .- Float64.(s_val)))
  end

  MOI.set(model, MOI.LazyConstraintCallback(), _lazy)

  optimize!(model)

  sol = value.(s) .> 0.5
  chosen = findall(sol)
  return chosen, value.(s)[chosen]
end
