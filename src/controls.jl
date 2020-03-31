function hebiMPPI(env_tconstructor = n -> Tuple(HebiPickup() for _=1:n))
    env = first(env_tconstructor(1))
    a = getaction(env)
    sigma = zeros(7) #TODO
    sigma[1:3] .= 0.15
    sigma[4:6] .= 0.15
    sigma[end] = 0.05
    mppi = MPPI(
        env_tconstructor = env_tconstructor,
        covar = Diagonal(sigma .^ 2),
        lambda = 0.05,

        H = 50,
        K = 16,
        gamma = 1.0,
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

function hebi_jac(env::HebiPickup)
    # NOTE: Frames are noted by the following suffixes: w = world, c = chopstick, t = target
    posweight = 0.7
    rotweight = 1 - posweight

    jacpT = zeros(env.sim.m.nv, 3)
    jacrT = zeros(env.sim.m.nv, 3)
    jacp = transpose(jacpT)
    jacr = transpose(jacrT)
    @warn env.sim.m.nv

    i = 0
    function ctrlfn(env)
        @unpack m, d, mn, dn = env.sim

        MuJoCo.MJCore.mj_jacSite(m, d, vec(jacpT), vec(jacrT), 0) # TODO not 1-based indexing??
        jac = vcat(jacp, jacr)[:, 1:6]
        jacT = transpose(jac)

        dpos = (dn.site_xpos[:, :pose] - dn.site_xpos[:, :stick])
        #center = chopstickcenter(env)
        #dpos = (dn.geom_xpos[:, end] - center)

        wRp = RotMatrix(dn.site_xmat[:, :pose]...)'
        wRc = RotMatrix(dn.site_xmat[:, :stick]...)'
        wRcp = wRp * transpose(wRc)
        #wRct = flat * transpose(wRc)
        #drot = rotation_axis(wRct) * rotation_angle(wRct)
        drot = rotation_axis(wRcp) * rotation_angle(wRcp)

        #clamp!(dpos, -0.3, 0.3)
        #clampnorm!(dpos, 0.15)

        error = vcat(posweight * dpos, rotweight * drot)

        dtheta = zeros(6)
        sdls_jacctrl!(dtheta, jac, error, lammax=1)
        #dls_jacctrl!(dtheta, jac, error, lambda = 0.1)
        #pinv_jacctrl!(dtheta, jac, error)

        posgain = [40, 30, 30, 30, 30, 30]
        velgain = [40, 30, 30, 30, 30, 30] .* 0.5
        accgain = -ones(6) .* 0.1

        clamp!(dtheta, -0.25, 0.25)

        #ctrl = velgain .* -(d.qvel[1:6] - dtheta) #.+ accgain .* d.qacc[1:6]
        ctrl = posgain .* dtheta .+ -2.5 * d.qvel[1:6]
        clamp!(ctrl, -1.0, 1.0)
        a = getaction(env)
        a[1:6] .= ctrl

        setaction!(env, a)
        #env.sim.d.ctrl[end] = clamp(env.sim.d.ctrl[end], -100000, 100000)
        #@info "chopstick" env.sim.d.ctrl[end]

        forward!(env.sim)
    end
    visualize(env, HEBIManualChop(ctrlfn))
end

