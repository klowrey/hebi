using LinearAlgebra
using LyceumAI, LyceumBase, LyceumMuJoCo, LyceumMuJoCoViz, Shapes
using LyceumBase.Tools
using UniversalLogger
using Base: @propagate_inbounds
using Random
import LyceumBase: tconstruct, _rollout
using Statistics, StaticArrays, UnsafeArrays
using Distributions, Distances
using MuJoCo

using Optim
using LineSearches

include("hebi.jl")

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


## TODO some kind of struct to map from vector to fields and structs?

function randset!(h2, h1)
    h2.sim.m.dof_damping[1:3]      .= h1.sim.m.dof_damping[1] .+ rand(Uniform(-0.05, 4.5), 3)
    h2.sim.m.dof_damping[4:7]      .= h1.sim.m.dof_damping[4] .+ rand(Uniform(-0.05, 4.5), 4)
    h2.sim.m.dof_armature[1:3]     .= h1.sim.m.dof_armature[1] .+ rand(Uniform(0, 10.2), 3)
    h2.sim.m.dof_armature[4:7]     .= h1.sim.m.dof_armature[4] .+ rand(Uniform(0, 10.2), 4)
    h2.sim.m.dof_frictionloss[1:3] .= h1.sim.m.dof_frictionloss[1] .+ rand(Uniform(-0.05, 3.5), 3)
    h2.sim.m.dof_frictionloss[4:7] .= h1.sim.m.dof_frictionloss[4] .+ rand(Uniform(-0.05, 3.5), 4)
    h2
end
function resetset!(h2, h1)
    h2.sim.m.dof_damping      .= h1.sim.m.dof_damping
    h2.sim.m.dof_armature     .= h1.sim.m.dof_armature
    h2.sim.m.dof_frictionloss .= h1.sim.m.dof_frictionloss
end

function gradsysid(ref::HebiPickup, test::HebiPickup, ctrls)
    reset!(ref);  traj = _rollout(ref, ctrls)
    #reset!(test); data = _rollout(test, ctrls)

    #randset!(test, ref)
    #println(test.sim.m.dof_damping[1:7])
    #println(test.sim.m.dof_armature[1:7])
    #println(test.sim.m.dof_frictionloss[1:7])

    #modelvars = MultiShape(damping = VectorShape(Float64, 7), # only for joints
    #                       armature = VectorShape(Float64, 7),
    #                       frictionloss = VectorShape(Float64, 7))
    #function setparams!(env, p::AbstractVector)
    #    mp = modelvars(p)
    #    env.sim.m.dof_damping[1:7]  .= mp.damping
    #    env.sim.m.dof_armature[1:7] .= mp.armature
    #    env.sim.m.dof_frictionloss[1:7] .= mp.frictionloss
    #    env
    #end
    #function getparams!(p::AbstractVector, env)
    #    mp = modelvars(p)
    #    mp.damping      .= env.sim.m.dof_damping[1:7]
    #    mp.armature     .= env.sim.m.dof_armature[1:7]
    #    mp.frictionloss .= env.sim.m.dof_frictionloss[1:7]
    #    env
    #end

    resetset!(test, ref)
    test.sim.m.dof_damping[1] = 20.0
    modelvars = MultiShape(damping = ScalarShape(Float64))
    function setparams!(env, p::AbstractVector)
        mp = modelvars(p)
        env.sim.m.dof_damping[1]  = mp.damping
        env
    end
    function getparams!(p::AbstractVector, env)
        mp = modelvars(p)
        mp.damping = env.sim.m.dof_damping[1]
        env
    end
    osp = obsspace(ref)
    s = length(o.qpos) + length(o.qvel)
    function opt(P, env)
        if any(x->x < 0.0, P) # no negative params
            return 1e9
        end
        setparams!(env, P)

        reset!(env)
        roll = _rollout(env, ctrls)

        #return mse(roll.obses, traj.obses) # from LyceumAI
        return mse(roll.obses[1:s,:], traj.obses[1:s,:]) # from LyceumAI
    end

    tests = [ HebiPickup() for _=1:Threads.nthreads() ]
    N = length(modelvars)
    space = [ zeros(N) for i=1:N ]
    function optgrad!(storage, P)
        for i=1:N
            space[i] .= P
            space[i][i] += 1e-4
        end
        z = opt(P, test)
        
        Threads.@threads for i=1:N
            tid = Threads.threadid() 
            storage[i] = (opt(space[i], tests[tid]) - z) / 1e-4
        end
    end

    initP = allocate(modelvars)
    getparams!(initP, test)

    options = Optim.Options(allow_f_increases=false,
                            show_trace=true, show_every=10,
                            iterations=40000, time_limit=60*10)
    result = optimize(p->opt(p, test), optgrad!, initP, LBFGS(), options)
    setparams!(test, result.minimizer)
    return result
end

function testsysid(ref::HebiPickup, test::HebiPickup, ctrls; optm=:NM)
    reset!(ref);  traj = _rollout(ref, ctrls)
    #reset!(test); data = _rollout(test, ctrls)

    randset!(test, ref)
    println(test.sim.m.dof_damping[1:7])
    println(test.sim.m.dof_armature[1:7])
    println(test.sim.m.dof_frictionloss[1:7])
    modelvars = MultiShape(x8_damping      = ScalarShape(Float64),
                           x5_damping      = ScalarShape(Float64),
                           x8_armature     = ScalarShape(Float64),
                           x5_armature     = ScalarShape(Float64),
                           x8_frictionloss = ScalarShape(Float64),
                           x5_frictionloss = ScalarShape(Float64))
    function setparams!(env, p::AbstractVector)
        mp = modelvars(p)
        env.sim.m.dof_damping[1:3]      .= mp.x8_damping
        env.sim.m.dof_damping[4:7]      .= mp.x5_damping
        env.sim.m.dof_armature[1:3]     .= mp.x8_armature
        env.sim.m.dof_armature[4:7]     .= mp.x5_armature
        env.sim.m.dof_frictionloss[1:3] .= mp.x8_frictionloss
        env.sim.m.dof_frictionloss[4:7] .= mp.x5_frictionloss
        env
    end
    function getparams!(p::AbstractVector, env)
        mp = modelvars(p)
        mp.x8_damping      = env.sim.m.dof_damping[1]
        mp.x5_damping      = env.sim.m.dof_damping[4]
        mp.x8_armature     = env.sim.m.dof_armature[1]
        mp.x5_armature     = env.sim.m.dof_armature[4]
        mp.x8_frictionloss = env.sim.m.dof_frictionloss[1]
        mp.x5_frictionloss = env.sim.m.dof_frictionloss[4]
        env
    end

    # One var testing
    # resetset!(test, ref)
    # test.sim.m.dof_damping[2] = 20.0
    # test.sim.m.dof_damping[3] = 20.0
    # modelvars = MultiShape(damping = VectorShape(Float64, 2))
    # function setparams!(env, p::AbstractVector)
    #     mp = modelvars(p)
    #     env.sim.m.dof_damping[2]  = mp.damping[1]
    #     env.sim.m.dof_damping[3]  = mp.damping[2]
    #     env
    # end
    # function getparams!(p::AbstractVector, env)
    #     mp = modelvars(p)
    #     mp.damping[1] = env.sim.m.dof_damping[2]
    #     mp.damping[2] = env.sim.m.dof_damping[3]
    #     env
    # end

    osp = obsspace(ref)
    s = length(osp.qpos) + length(osp.qvel)

    function opt(P, env=test)
        #if any(x->x < 0.0, P) # no negative params
        #    return sum(traj.obses .^ 2) * 1 // length(traj.obses)
        #end
        setparams!(env, P)

        reset!(env)
        roll = _rollout(env, ctrls)

        return mse(roll.obses[1:s,:], traj.obses[1:s,:]) # from LyceumAI
    end

    tests = [ HebiPickup() for _=1:Threads.nthreads() ]
    N = length(modelvars)
    space = [ zeros(N) for i=1:N ]
    function optgrad!(storage, P)
        for i=1:N
            space[i] .= P
            space[i][i] += 1e-4
        end
        z = opt(P, test)
        
        Threads.@threads for i=1:N
            tid = Threads.threadid() 
            storage[i] = (opt(space[i], tests[tid]) - z) / 1e-4
        end
    end

    initP = allocate(modelvars)
    getparams!(initP, test)
    println(initP)
    clamp!(initP, 0.001, 4.9)
    lower = fill(0.0001, N) # approx upper and lower bounds? can specialize more...
    upper = fill(5.0, N)

    options = Optim.Options(allow_f_increases=false,
                            show_trace=true, show_every=10,
                            iterations=40000, time_limit=60*10)
    #result = optimize(opt, initP, NelderMead(; initial_simplex=MySimplexer{T}(xmax, 0.0)), options)
    if optm == :NM
        #result = optimize(opt, initP, NelderMead(), options)
        result = optimize(opt, lower, upper, initP, Fminbox(NelderMead()), options)
    elseif optm == :LBFGS
        #result = optimize(p->opt(p, test), optgrad!, initP, LBFGS(linesearch = LineSearches.BackTracking()), options)
        #result = optimize(p->opt(p, test), initP, LBFGS(), options)
        #result = optimize(opt, lower, upper, initP, Fminbox(LBFGS()), options)
        result = optimize(p->opt(p, test), optgrad!, lower, upper, initP,
                          Fminbox(LBFGS(linesearch = LineSearches.BackTracking())), options)
    elseif optm == :PS
        #all_minx = [Float64[] for i in Threads.nthreads()]
        #all_minf = [Float64[] for i in Threads.nthreads()]
        #Threads.@threads for i in 1:N 
        #    (minf,minx,ret) = optimize(opt, rand(2))
        #    push!(all_minx[Threads.threadid()],minx)
        #    push!(all_minf[Threads.threadid()],minf)
        #end

        #minxs = vcat(all_minx...)
        #minfs = vcat(all_minf...)
        result = optimize(opt, initP,
                          ParticleSwarm(lower=lower, upper=upper, n_particles=60), options)
    else
        @warn "optimizer $optm not configured yet."
    end
    setparams!(test, result.minimizer)

    return result

end

function sinfn(env)
    m, d = env.sim.m, env.sim.d
    t = 2 * d.time

    d.ctrl[1] = sin(t)
    d.ctrl[2] = 1.0 + 0.25 * sin(t)
    d.ctrl[3] = 0.6 + 0.25 * sin(2t)
    d.ctrl[4] = sin(t)
    d.ctrl[5] = 0.1 * sin(2t)
    #d.ctrl[6] = sin(t)
    d.ctrl[7] = -0.21 + 0.05 * sin(t)

    #d.ctrl[9]  = 1.0   # when using position control
    #d.ctrl[10] = 1.0
    #d.ctrl[14] = -0.21
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

function vizdif(env::HebiPickup, env2::HebiPickup,
                ctrls::AbstractMatrix)
    reset!(env);  traj = _rollout(env, ctrls)
    reset!(env2); data = _rollout(env2, ctrls)

    println(mse(data.obses, traj.obses))
    visualize(env, trajectories=[traj.states, data.states])
end

function jacctrl(env::HebiPickup)
    jacp = zeros(env.sim.m.nv, 3)
    jacr = zeros(env.sim.m.nv, 3)

    function ctrlfn(env)
        m, d = env.sim.m, env.sim.d
        MuJoCo.MJCore.mj_jacSite(m, d, vec(jacp), vec(jacr), 0) # not 1-based indexing??

        # have the object as the target site
        delta = SPoint3D(d.geom_xpos, m.ngeom) - SPoint3D(d.site_xpos, 1)

        #d.ctrl[1:7] .= jacp[:,1:7]' * delta
        d.ctrl[1:7] .= 15.0 .* jacp[1:7, :] * delta
        d.ctrl[9]  = 1.0   # when using position control
        d.ctrl[10] = 1.0
        d.ctrl[14] = -0.21
    end
    visualize(env, controller=ctrlfn)
end

function viz_mppi(mppi::MPPI, env::HebiPickup)
    a = allocate(actionspace(env))
    o = allocate(obsspace(env))
    s = allocate(statespace(env))
    function ctrlfn(env) 
        getstate!(s, env)
        getaction!(a, s, o, mppi)
        setaction!(env, a)
    end
    visualize(env, controller=ctrlfn)
end


