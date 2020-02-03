
function hebiMPPI(etype = HebiPickup; T = 100, H = 32, K = 24, lambda=0.5)
    env = etype()

    ## The following parameters work well for this get-up task, and may work for
    ## similar tasks, but are not invariant to the model.
    a = getaction(env)
    a[1:7]   .= 0.2
    a[8:end] .= 0.0001

    mppi = MPPI(
        env_tconstructor = n -> tconstruct(etype, n),
        covar = Diagonal(a.^2),
        lambda = lambda,
        H = H,
        K = K,
        gamma = 1.0,
    )

    #iter = ControllerIterator(mppi, env; T = T, plotiter = div(T, 2), randstart = true)

    ### We can time the following loop; if it ends up less than the time the
    ### MuJoCo models integrated forward in, then one could conceivably run this
    ### MPPI MPC controller interactively...
    #elapsed = @elapsed for (t, traj) in iter
    #    ## If desired, one can inspect `traj`, `env`, or `mppi` at each timestep.
    #end

    #println("$elapsed / $(time(env)) : $(elapsed / time(env))")
    #if elapsed < time(env)
    #    @info "We ran in real time!"
    #end

    return mppi, env #, iter.trajectory
end

function getsincontrols(env, T=2000)
    reset!(env)
    actions = zeros(length(actionspace(env)), T)
    for t=1:T
        sinfn(env) # set controls
        getaction!(view(actions, :, t), h) # save actions
        step!(env)
    end
    reset!(env)
    return actions
end

function sinctrl(env::HebiPickup)
    visualize(env, controller=sinfn)
end

function jacctrl(env::HebiPickup, gain=100.0)
    jacp = zeros(env.sim.m.nv, 3)
    jac  = zeros(env.sim.m.nu, 3)
    jacr = zeros(env.sim.m.nv, 3)
    ctrl = zeros(env.sim.m.nu)
    copyidx = CartesianIndices((env.sim.m.nu,3))

    function ctrlfn(env)
        m, d = env.sim.m, env.sim.d
        MuJoCo.MJCore.mj_jacSite(m, d, vec(jacp), vec(jacr), 0) # not 1-based indexing??
        #MuJoCo.MJCore.mj_jacBodyCom(m, d, vec(jacp), vec(jacr), 8) # not 1-based indexing??
        #display(jacr)
        #copyto!(jac, copyidx, jacp, copyidx) # jac .= jac[1:mu, :] # err not working?

        # have the object as the target site through it's body_xpos
        positiondelta = gain .* (SPoint3D(d.xpos, m.nbody) - SPoint3D(d.site_xpos, 1))

        # already transposed as mujoco is row-major; julia is col-major
        ctrl .= jacp[1:7,:] * positiondelta
        clamp!(ctrl, -1.0, 1.0)
        #display(ctrl)

        setaction!(env, ctrl)

        forward!(env.sim)

        #d.ctrl[9]  = 1.0   # when using position control
        #d.ctrl[10] = 1.0
        #d.ctrl[14] = -0.21
    end
    visualize(env, controller=ctrlfn)
end
