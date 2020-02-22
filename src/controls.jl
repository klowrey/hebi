function hebiMPPI(etype = HebiPickup; T = 100, H = 32, K = 24, lambda=0.5)
    env = etype()

    a = getaction(env)
    #a[1:7]   .= 0.2
    #a[8:end] .= 0.0001

    osp = obsspace(env)
    value = @closure (o) -> begin
        osh = osp(o)
        -100 * norm(osh.dp) - 10 * norm(osh.dr)
    end

    a[1:7]   .= 0.2
    a[8:end] .= 0.0001
    mppi = MPPI(
        env_tconstructor = n -> tconstruct(etype, n),
        #covar = Diagonal(a.^2),
        covar = Diagonal(0.7^2*I, length(actionspace(env))),
        lambda = 0.8,
        H = 50,
        K = 16,
        gamma = 1.0,
        value = value
    )

    a = allocate(actionspace(env))
    o = allocate(obsspace(env))
    s = allocate(statespace(env))

    ctrlfn = @closure env -> begin
        getstate!(s, env)
        getaction!(a, s, mppi)
        setaction!(env, a)
    end

    visualize(env, controller=ctrlfn)
end

