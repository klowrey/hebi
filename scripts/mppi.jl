include("common/common.jl")

function viz_mppi()
    env = HEBI.HEBIPickup(action_mode = :efc_vel)

    sigma = zeros(actionspace(env))
    @assert length(sigma) == 7
    sigma[1:3] .= 0.15
    sigma[4:6] .= 0.15
    sigma[end] = 0.05

    mppi = MPPI(
        env_tconstructor = n -> Tuple(HEBI.HEBIPickup(action_mode = :efc_vel) for _ = 1:n),
        covar = Diagonal(sigma .^ 2),
        lambda = 0.05,
        H = 50,
        K = 16,
        gamma = 1.0,
    )

    a = Array(undef, actionspace(env))
    o = Array(undef, obsspace(env))
    s = Array(undef, statespace(env))
    ctrlfn = @closure env -> begin
        getstate!(s, env)
        getaction!(a, s, mppi)
        setaction!(env, a)
    end

    visualize(env, controller = ctrlfn)
end
