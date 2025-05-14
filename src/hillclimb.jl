# simple “warm‐start” hill‐climber to pick a good support
function portfolios_hillclimb(μ, Σ, A, l, u, mininv, k::Int, inds0::Vector{Int};
                              maxiter::Int=50,
                              SoRCoeff::Float64=1.7)
    n = length(μ)
    f = size(Σ,1)

    # initialize
    α = zeros(f)
    λ = 0.0
    βl = zeros(size(A,1));  βu = copy(βl)
    ρ  = zeros(n)
    inds = copy(inds0)
    iter = 0

    while iter < maxiter
      iter += 1
      old = copy(inds)
      D = inner_dual(μ,Σ,A,l,u,mininv,inds)
      # aggregate duals into full ρ
      ρ_full = zeros(n);  ρ_full[inds] .= D.ρ

      # running average
      α  .= ((iter-SoRCoeff)*α  + SoRCoeff*D.α)  / iter
      λ  =  ((iter-SoRCoeff)*λ  + SoRCoeff*D.λ)  / iter
      βl .= ((iter-SoRCoeff)*βl + SoRCoeff*D.βl) / iter
      βu .= ((iter-SoRCoeff)*βu + SoRCoeff*D.βu) / iter
      ρ  .= ((iter-SoRCoeff)*ρ  + SoRCoeff*ρ_full) / iter

      # discrete descent
      #  p_i = -(γ_i/2)*(...)^2 + mininv_i*ρ_i  -- here γ=1
      p = -(α'*(Σ[:,inds]).^2)/2 .+ mininv[inds].*ρ_full[inds]
      # rank assets by |x - p| heuristic
      scores = abs.(Σ[:,inds]'*α .- p)
      perm   = sortperm(scores, rev=true)[1:k]
      inds   = inds[perm]

      # break if no change
      if inds == old
        break
      end
    end

    return inds
end
