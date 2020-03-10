function dls_jacctrl!(dtheta::AbsVec, jac::AbsMat, error::AbsVec; lambda::Real = 1.1)
    dtheta .= inv((jac'*jac + lambda^2 * I)) * jac' * error
end

pinv_jacctrl!(dtheta::AbsVec, jac::AbsMat, error::AbsVec) = dtheta .= pinv(jac) * error

function transpose_jacctrl!(dtheta::AbsVec, jac::AbsMat, error::AbsVec)
    dtheta .= transpose(jac) * error
end

function sdls_jacctrl!(dtheta::AbsVec, jac::AbsMat, error::AbsVec; lammax::Real = pi/8)
    dofefc, nv = size(jac)
    u, s, v = svd(jac)

    N = zeros(dofefc)
    for i=1:dofefc
        N[i] = norm(u[:, i])
    end

    M = zeros(dofefc)
    for i=1:dofefc
        for j=1:nv
            pj = norm(jac[:, j])
            M[i] += 1/s[i] * abs(v[j, i]) * pj
        end
    end

    lam = zeros(dofefc)
    for i=1:dofefc
        lam[i] = min(1, N[i] / M[i]) * lammax
    end

    phi = zeros(nv, dofefc)
    for i=1:dofefc, j=1:nv
        alpha = transpose(u[:, i]) * error
        phi[:, i] = clampnorm!(1/s[i] * alpha * v[:, i], lam[i], 1)
        #phi[:, i] = clampmaxabs!(1/s[i] * alpha * v[:, i], lam[i])
    end

    for i=1:dofefc
        dtheta .+= phi[:, i]
    end
    clampnorm!(dtheta, lammax, 1)
    #clampmaxabs!(dtheta, lammax)

    return dtheta
end


wxyz(q::Quat{T}) where {T} = SVector{4,T}(q.w, q.x, q.y, q.z)
wxyz(r::Rotation{3}) = wxyz(Quat(r))

function jacctrl(env::HebiPickup)
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

function opttest4()
    env = HebiPickup()
    reset!(env)

    jacpT = zeros(env.sim.m.nv, 3)
    jacrT = zeros(env.sim.m.nv, 3)
    jacp = transpose(jacpT)
    jacr = transpose(jacrT)

    q0 = SVector{6}(env.sim.d.qpos[1:6])

    posweight = 1
    rotweight = 1

    maxdist = pi
    maxvel = 0.15
    lower = -maxvel * ones(6)
    upper = maxvel * ones(6)

    velgains = SVector{6, Float64}([40, 30, 30, 30, 30, 30] )
    accgains = SVector{6, Float64}([1, 1, 1, 1, 1, 1])

    # Frames: w = world, c = chopstick, t = target
    i = 0
    function ctrlfn(env)
        i += 1
        @unpack m, d, mn, dn = env.sim


        MuJoCo.MJCore.mj_jacSite(m, d, vec(jacpT), vec(jacrT), 0) # not 1-based indexing??

        dpos = (dn.site_xpos[:, :target] - dn.site_xpos[:, :chopstick])
        # so that position errors and orientation errors are of same magnitude
        dist = norm(dpos)
        dpos = (dpos / dist) * min(dist, maxdist)

        wRt = RotMatrix(dn.site_xmat[:, :target]...)'
        wRc = RotMatrix(dn.site_xmat[:, :chopstick]...)'
        wRct = wRt * transpose(wRc)
        drot = rotation_axis(wRct) * rotation_angle(wRct)

        twist = SVector{6}(vcat(posweight * dpos, rotweight * drot))
        jac = SMatrix{6,6}(vcat(jacp, jacr)[:, 1:6])
        jacT = transpose(jac)
        qnow = SVector{6}(d.qpos[1:6])

        f! = @closure (error, dtheta) -> begin
            error .= (jac * dtheta - twist)
        end

        results = optimize!(
            #LeastSquaresProblem(x = zeros(6), f! = f!, g! = j!, output_length = 6),
            LeastSquaresProblem(x = zeros(6), f! = f!, output_length = 6),
            Dogleg(),
            lower = lower,
            upper = upper,
        )
        dtheta = results.minimizer

        #if mod(i, 50) == 0
        #    display(results)
        #end
        @info "dtheta" dtheta

        clamp!(dtheta, -maxvel, maxvel)

        ctrl = velgains .* dtheta
        clamp!(ctrl, -1.0, 1.0)

        a = getaction(env)
        a[1:6] .= ctrl
        setaction!(env, a)
    end

    visualize(env, HEBIManualChop(ctrlfn))
    return nothing
end

function opttest5()
    env = HebiPickup()
    model = HebiPickup()
    reset!(env)
    reset!(model)

    jacpT = zeros(env.sim.m.nv, 3)
    jacrT = zeros(env.sim.m.nv, 3)
    jacp = transpose(jacpT)
    jacr = transpose(jacrT)

    q0 = SVector{6}(env.sim.d.qpos[1:6])

    @unpack m, d, mn, dn = model.sim
    posweight = 0.999
    rotweight = 1 - posweight

    maxdist = pi
    maxvel = 0.15
    lower = -maxvel * ones(6)
    upper = maxvel * ones(6)

    velgains = SVector{6, Float64}([40, 30, 30, 30, 30, 30])
    accgains = SVector{6, Float64}([1, 1, 1, 1, 1, 1])

    f! = (error, dtheta) -> begin
        d.qpos[1:6] .+= dtheta
        MJCore.mj_kinematics(m, d)

        dpos = (dn.site_xpos[:, :target] - dn.site_xpos[:, :chopstick])
        wRt = RotMatrix(dn.site_xmat[:, :target]...)'
        wRc = RotMatrix(dn.site_xmat[:, :chopstick]...)'
        wRct = wRt * transpose(wRc)
        drot = MVector(rotation_axis(wRct) * rotation_angle(wRct))

        twist = vcat(dpos, drot)

        error .= twist

        d.qpos[1:6] .-= dtheta

        return error
    end


    f2 = dtheta -> norm(f!(zeros(6), dtheta))

    i = 0
    ctrlfn = @closure env -> begin
        i += 1
        # error = K(qpos) - pose0
        setstate!(model, getstate(env))

        #results = optimize(f2, lower, upper, zeros(6), Fminbox(opt), Optim.Options(time_limit = 10*timestep(env)))
        #results = optimize(f, zeros(6), LBFGS())
        results = optimize!(
            LeastSquaresProblem(x = zeros(6), f! = f!, output_length = 6, autodiff = :central),
            Dogleg(),
            #lower = lower,
            #upper = upper,
            #x_tol = 0.0,
            #f_tol = 0.0,
            #g_tol = 0.0,
        )

        dqpos = results.minimizer

        if mod(i, 50) == 0
            display(results)
        end

        posgain = [20, 20, 30, 30, 30, 30]
        velgain = [40, 30, 30, 30, 30, 30] .* 0.5
        accgain = -ones(6) .* 0.1

        clamp!(dqpos, -maxvel, maxvel)

        ctrl = posgain .* dqpos - 5 * env.sim.d.qvel[1:6]
        clamp!(ctrl, -1.0, 1.0)
        a = getaction(env)
        a[1:6] .= ctrl
        setaction!(env, a)
    end

    visualize(env, HEBIManualChop(ctrlfn))
    return nothing
end
