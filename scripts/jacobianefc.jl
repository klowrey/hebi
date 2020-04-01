include("common/common.jl")

function viz_jacobianefc()
    env = HEBI.HEBIPickup(action_mode = :joint_torque)

    @assert length(actionspace(env)) == 7
    T = eltype(actionspace(env))

    posweight = T(0.7)
    rotweight = T(1 - posweight)

    jacpT = zeros(T, env.sim.m.nv, 3)
    jacrT = zeros(T, env.sim.m.nv, 3)
    jacp = transpose(jacpT)
    jacr = transpose(jacrT)
    dtheta = zeros(T, 6)
    action = zeros(actionspace(env))

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

        error = vcat(posweight * dpos, rotweight * drot)

        HEBI.sdls_jacctrl!(dtheta, jac, error, lammax = 1)

        getaction!(action, env) # we don't want to overwrite the chopstick command
        clamp!(dtheta, -0.25, 0.25)
        action[1:6] .= 40 .* dtheta .+ -2.5 .* d.qvel[1:6]

        setaction!(env, action)
    end

    visualize(env, HEBIManualChop(ctrlfn))
end
