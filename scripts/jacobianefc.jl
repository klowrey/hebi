include("common.jl")

function viz_jacobianefc()
    env = HEBI.HEBIPickup(action_mode=:joint_torque)

     # NOTE: Frames are noted by the following suffixes: w = world, c = chopstick, p = pose # TODO move this

    posweight = 0.7
    rotweight = 1 - posweight

    jacpT = zeros(env.sim.m.nv, 3)
    jacrT = zeros(env.sim.m.nv, 3)
    jacp = transpose(jacpT)
    jacr = transpose(jacrT)

    i = 0
    ctrlfn = @closure env -> begin
        @unpack m, d, mn, dn = env.sim

        MuJoCo.MJCore.mj_jacSite(m, d, vec(jacpT), vec(jacrT), 0) # TODO not 1-based indexing??
        jac = vcat(jacp, jacr)[:, 1:6]
        jacT = transpose(jac)

        dpos = (dn.site_xpos[:, :pose] - dn.site_xpos[:, :stick])

        wRp = RotMatrix(dn.site_xmat[:, :pose]...)'
        wRc = RotMatrix(dn.site_xmat[:, :stick]...)'
        wRcp = wRp * transpose(wRc)
        drot = rotation_axis(wRcp) * rotation_angle(wRcp)

        #clamp!(dpos, -0.3, 0.3)
        #clampnorm!(dpos, 0.15)

        error = vcat(posweight * dpos, rotweight * drot)

        dtheta = zeros(6)
        HEBI.sdls_jacctrl!(dtheta, jac, error, lammax = 1)
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

        forward!(env.sim) # TODO
    end

    visualize(env, controller = ctrlfn)
end
