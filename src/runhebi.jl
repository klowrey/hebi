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

using BlackBoxOptim
using Distributed
using SharedArrays

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

function sysidfromfile(s, test::HebiPickup; optm=:NM, batch=nothing)
    dt, pos, vel, act = datafile(s)

    if batch == nothing
        batch = 1:size(pos,2)
    end
    return filesysid(vcat(pos, vel), test, act; optm = optm, batch=batch)
end

function datafile(s="example.csv", nv=7)
    d = CSV.read(s)
    println(names(d))
    r, c = size(d)

    pos = zeros(nv, r)
    vel = zeros(nv, r)
    act = zeros(nv, r)
    eff = zeros(nv, r)

    @time Threads.@threads for i=1:r
        pos[:, i] .= tryparse.(Float64, split(d.positions[i]))
        vel[:, i] .= tryparse.(Float64, split(d.velocities[i]))
        act[:, i] .= tryparse.(Float64, split(d.pwmCommands[i]))
        eff[:, i] .= tryparse.(Float64, split(d.efforts[i]))
    end
    
    println(r)
    dt = (d.timestamp_us .- d.timestamp_us[1]) * 1e-6

    # to currently match the 0.004 DT model (dt = 0.002 with skip = 2)
    #return dt[2:4:end], pos[:,2:4:end], vel[:,2:4:end], act[:,2:4:end]
    return dt, pos, vel, act, eff 
end

function filesysid(refstate, test::HebiPickup, ctrls; optm=:NM, batch = 1:size(refstate, 2))
    #modelvars = MultiShape(x8_damping      = ScalarShape(Float64),
    #                       x5_damping      = ScalarShape(Float64),
    #                       x8_armature     = ScalarShape(Float64),
    #                       x5_armature     = ScalarShape(Float64),
    #                       x8_frictionloss = ScalarShape(Float64),
    #                       x5_frictionloss = ScalarShape(Float64),
    #                       x8_gear         = ScalarShape(Float64),
    #                       x8_16_gear      = ScalarShape(Float64),
    #                       x5_gear         = ScalarShape(Float64),
    #                      )
    modelvars = MultiShape(x8_damping      = ScalarShape(Float64),
                           x8_armature     = ScalarShape(Float64),
                           x8_frictionloss = ScalarShape(Float64),
                           x8_gear         = ScalarShape(Float64),
                           x8_16_damping      = ScalarShape(Float64),
                           x8_16_armature     = ScalarShape(Float64),
                           x8_16_frictionloss = ScalarShape(Float64),
                           x8_16_gear      = ScalarShape(Float64),
                          )
    #modelvars = MultiShape(x5_damping      = ScalarShape(Float64),
    #                       x5_armature     = ScalarShape(Float64),
    #                       x5_frictionloss = ScalarShape(Float64),
    #                       x5_gear         = ScalarShape(Float64),
    #                      )
    scalingfactor = ones(6) #[ 100.0, 10.0,
                    #    1.0, 1.0,
                    #   10.0, 10.0
                    #]
    N = length(modelvars)
    function setparams!(env, p::AbstractVector, varspace)
        mp = varspace(p)
        m = env.sim.m
        ##env.sim.m.dof_damping[1:3]      .= scalingfactor[1] * mp.x8_damping
        #m.dof_damping[1]      = scalingfactor[1] * mp.x8_damping
        #m.dof_damping[2]      = 2 * scalingfactor[1] * mp.x8_damping
        #m.dof_damping[3]      = scalingfactor[1] * mp.x8_damping
        ##m.dof_damping[4:7]      .= scalingfactor[2] * mp.x5_damping
        #m.dof_armature[1:3]     .= scalingfactor[3] * mp.x8_armature
        ##m.dof_armature[4:7]     .= scalingfactor[4] * mp.x5_armature
        #m.dof_frictionloss[1:3] .= scalingfactor[5] * mp.x8_frictionloss
        ##m.dof_frictionloss[4:7] .= scalingfactor[6] * mp.x5_frictionloss

        m.dof_damping[1]      = mp.x8_damping
        m.dof_damping[2]      = mp.x8_16_damping
        m.dof_damping[3]      = mp.x8_damping
        m.dof_armature[1]     = mp.x8_armature
        m.dof_armature[2]     = mp.x8_16_armature
        m.dof_armature[3]     = mp.x8_armature
        m.dof_frictionloss[1] = mp.x8_frictionloss
        m.dof_frictionloss[2] = mp.x8_16_frictionloss
        m.dof_frictionloss[3] = mp.x8_frictionloss

        m.actuator_gear[1,1]    = mp.x8_gear
        m.actuator_gear[1,2]    = mp.x8_16_gear
        m.actuator_gear[1,3]    = mp.x8_gear
        #m.actuator_gear[1,4:7] .= mp.x5_gear

        env
    end
    function getparams!(p::AbstractVector, env, varspace)
        mp = varspace(p)
        m = env.sim.m
        #mp.x8_damping      = m.dof_damping[1]      / scalingfactor[1] 
        ##mp.x5_damping      = m.dof_damping[4]      / scalingfactor[2]
        #mp.x8_armature     = m.dof_armature[1]     / scalingfactor[3]
        ##mp.x5_armature     = m.dof_armature[4]     / scalingfactor[4]
        #mp.x8_frictionloss = m.dof_frictionloss[1] / scalingfactor[5]
        ##mp.x5_frictionloss = m.dof_frictionloss[4] / scalingfactor[6]

        mp.x8_gear    = m.actuator_gear[1,1]
        mp.x8_16_gear = m.actuator_gear[1,2]
        #mp.x5_gear    = m.actuator_gear[1,4]

        mp.x8_damping         = m.dof_damping[1]      
        mp.x8_16_damping      = m.dof_damping[2]      
        mp.x8_armature        = m.dof_armature[1]     
        mp.x8_16_armature     = m.dof_armature[2]     
        mp.x8_frictionloss    = m.dof_frictionloss[1] 
        mp.x8_16_frictionloss = m.dof_frictionloss[2] 

        env
    end

    osp = obsspace(test)
    s = length(osp.qpos) + length(osp.qvel) #- 6 # ignore the object's data for now
    println(s)

    #batch = 1500:8000 #1000:6000
    nq = test.sim.m.nq
    nv = test.sim.m.nv

    function opt(P, env=test, Y=refstate, batch=batch)
        setparams!(env, P, modelvars)

        reset!(env)
        env.sim.d.qpos .= Y[1:7,  batch.start]
        env.sim.d.qvel .= Y[8:14, batch.start]
        env.sim.d.ctrl .= ctrls[:,batch.start]
        forward!(env.sim)
        roll = _rollout(env, ctrls[:,batch])

        #return mse(Y[:,batch], roll.obses[1:s,:]) # from LyceumAI
        idx = 1:3 #4:6
        vid = nq .+ idx
        return mse(Y[idx,batch], roll.obses[idx,:]) + mse(Y[vid,batch], roll.obses[vid,:])
        #return mse(Y[1:7,batch], roll.obses[1:7,:]) + mse(10 .* Y[8:14,batch], 10 .* roll.obses[8:14,:])
    end

    tests = [ HebiPickup() for _=1:Threads.nthreads() ] # independent models for parallel eval
    Peps = [ zeros(N) for i=1:N+1 ]
    function optgrad!(storage, P, ep=1e-4)
        for i=1:N
            Peps[i] .= P
            Peps[i][i] += ep 
        end
        Peps[end] .= P

        cache = zeros(N+1)
        Threads.@threads for i=1:(N+1)
            tid = Threads.threadid() 
            cache[i] = opt(Peps[i], tests[tid]) #- z) / ep
        end
        storage .= (cache[1:N] .- cache[end]) ./ ep
    end

    initP = allocate(modelvars)
    getparams!(initP, test, modelvars)
    println(initP)
    hi = 2.0
    lo = 0.001
    #Jj clamp!(initP, 2*lo, hi - 1e-4)
    lower = fill(lo, N) # approx upper and lower bounds? can specialize more...
    #upper = [50.0, 10.0, 5, 5, 5, 5, 100, 100] #fill(1.0, N)
    #upper = [10.0, 5, 5, 100] # 4:6
    upper = [100.0, 5, 5, 100, 100.0, 5, 5, 100] # 1:3

    options = Optim.Options(allow_f_increases=false,
                            show_trace=true, show_every=10,
                            iterations=40000, time_limit=60*60)
    if optm == :NM
        #result = optimize(opt, initP, NelderMead(; initial_simplex=MySimplexer{T}(xmax, 0.0)), options)
        result = optimize(opt, lower, upper, initP, Fminbox(NelderMead()), options) # probably need custom simplex initializer
        setparams!(test, result.minimizer, modelvars)
    elseif optm == :LBFGS
        #result = optimize(opt, lower, upper, initP, Fminbox(LBFGS()), options)
        result = optimize(p->opt(p, test), optgrad!,
                          lower, upper, initP,
                          Fminbox(LBFGS(alphaguess = LineSearches.InitialQuadratic(),
                                        linesearch = LineSearches.BackTracking())),
                          options)
        setparams!(test, result.minimizer, modelvars)
    elseif optm == :PS
        @warn "optimizer $optm not configured yet."
        #result = optimize(opt, initP,
        #                  ParticleSwarm(lower=lower, upper=upper, n_particles=18), options) # doesn't work?
    elseif optm == :BBO
        # probably needs much more parallel evals to be useful
        nworkers() == 1 && addprocs(Threads.nthreads())
        @info "using $(nworkers()) worker processes" 
        #sa = SharedArray{HebiPickup}(nworkers())
        #@everywhere popt(p) = opt(p, HebiPickup())
        bbopt = bbsetup(p->opt(p, test); Method=:xnes, SearchRange = (0.0001, 5.0),
                        NumDimensions = length(modelvars), MaxFuncEvals = 1000)#, Workers = workers())
        result = bboptimize(bbopt)
    else
        @warn "optimizer $optm not configured yet."
    end

    return result

end

function testsysid(ref::HebiPickup, test::HebiPickup, ctrls; optm=:NM)
    reset!(ref)
    traj = _rollout(ref, ctrls)

    randset!(test, ref)
    #println(test.sim.m.dof_damping[1:7])
    #println(test.sim.m.dof_armature[1:7])
    #println(test.sim.m.dof_frictionloss[1:7])
    modelvars = MultiShape(x8_damping      = ScalarShape(Float64),
                           x5_damping      = ScalarShape(Float64),
                           x8_armature     = ScalarShape(Float64),
                           x5_armature     = ScalarShape(Float64),
                           x8_frictionloss = ScalarShape(Float64),
                           x5_frictionloss = ScalarShape(Float64))
    scalingfactor = [ 1000.0, 100.0, 10.0, 10.0, 1000.0, 10.0 ]
    N = length(modelvars)
    function setparams!(env, p::AbstractVector)
        mp = modelvars(p)
        env.sim.m.dof_damping[1:3]      .= scalingfactor[1] * mp.x8_damping
        env.sim.m.dof_damping[4:7]      .= scalingfactor[2] * mp.x5_damping
        env.sim.m.dof_armature[1:3]     .= scalingfactor[3] * mp.x8_armature
        env.sim.m.dof_armature[4:7]     .= scalingfactor[4] * mp.x5_armature
        env.sim.m.dof_frictionloss[1:3] .= scalingfactor[5] * mp.x8_frictionloss
        env.sim.m.dof_frictionloss[4:7] .= scalingfactor[6] * mp.x5_frictionloss
        env
    end
    function getparams!(p::AbstractVector, env)
        mp = modelvars(p)
        mp.x8_damping      = env.sim.m.dof_damping[1]      / scalingfactor[1] 
        mp.x5_damping      = env.sim.m.dof_damping[4]      / scalingfactor[2]
        mp.x8_armature     = env.sim.m.dof_armature[1]     / scalingfactor[3]
        mp.x5_armature     = env.sim.m.dof_armature[4]     / scalingfactor[4]
        mp.x8_frictionloss = env.sim.m.dof_frictionloss[1] / scalingfactor[5]
        mp.x5_frictionloss = env.sim.m.dof_frictionloss[4] / scalingfactor[6]
        env
    end

    osp = obsspace(ref)
    s = length(osp.qpos) + length(osp.qvel) - 6 # ignore the object's data for now

    function opt(P, env=test, Y=traj.obses[1:s,:])
        setparams!(env, P)

        reset!(env)
        roll = _rollout(env, ctrls)

        return mse(Y, roll.obses[1:s,:]) # from LyceumAI
    end

    tests = [ HebiPickup() for _=1:Threads.nthreads() ] # independent models for parallel eval
    Peps = [ zeros(N) for i=1:N+1 ]
    function optgrad!(storage, P, ep=1e-4)
        for i=1:N
            Peps[i] .= P
            Peps[i][i] += ep 
        end
        Peps[end] .= P

        #z = opt(P, test)
        
        cache = zeros(N+1)
        Threads.@threads for i=1:(N+1)
            tid = Threads.threadid() 
            cache[i] = opt(Peps[i], tests[tid])
        end
        storage .= (cache[1:N] .- cache[end]) ./ ep
    end

    initP = allocate(modelvars)
    #getparams!(initP, test)
    getparams!(initP, ref)
    println(initP)
    initP .+= 0.01
    clamp!(initP, 0.001, 4.9)
    lower = fill(0.0001, N) # approx upper and lower bounds? can specialize more...
    upper = fill(5.0, N)

    options = Optim.Options(allow_f_increases=false,
                            show_trace=true, show_every=10,
                            iterations=40000, time_limit=60*10)
    if optm == :NM
        #result = optimize(opt, initP, NelderMead(; initial_simplex=MySimplexer{T}(xmax, 0.0)), options)
        result = optimize(opt, lower, upper, initP, Fminbox(NelderMead()), options) # probably need custom simplex initializer
        setparams!(test, result.minimizer)
    elseif optm == :LBFGS
        #result = optimize(opt, lower, upper, initP, Fminbox(LBFGS()), options)
        result = optimize(p->opt(p, test), optgrad!,
                          lower, upper, initP,
                          Fminbox(LBFGS(alphaguess = LineSearches.InitialQuadratic(),
                                        linesearch = LineSearches.BackTracking())),
                          options)
        setparams!(test, result.minimizer)
    elseif optm == :PS
        @warn "optimizer $optm not configured yet."
        #result = optimize(opt, initP,
        #                  ParticleSwarm(lower=lower, upper=upper, n_particles=18), options) # doesn't work?
    elseif optm == :BBO
        # probably needs much more parallel evals to be useful
        nworkers() == 1 && addprocs(Threads.nthreads())
        @info "using $(nworkers()) worker processes" 
        #sa = SharedArray{HebiPickup}(nworkers())
        #@everywhere popt(p) = opt(p, HebiPickup())
        bbopt = bbsetup(p->opt(p, test); Method=:xnes, SearchRange = (0.0001, 5.0),
                        NumDimensions = length(modelvars), MaxFuncEvals = 1000)#, Workers = workers())
        result = bboptimize(bbopt)
    else
        @warn "optimizer $optm not configured yet."
    end

    getparams!(initP, ref)
    println("Diff from reference model:")
    println(round.(initP .- result.minimizer, digits=3))
    println()

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

function vizdif_ref(env::HebiPickup, pos, vel, ctrls::AbstractMatrix)
    reset!(env)

    env.sim.d.qpos .= pos[:,1]
    env.sim.d.qvel .= vel[:,1]

    traj = _rollout(env, ctrls)
    ref = copy(traj.states)
    ref[2:8, :]  .= pos
    ref[9:15, :] .= vel
    ref[16:end, :] .= 0.0

    #reset!(env)
    visualize(env, trajectories=[ref, traj.states])
    return traj.states[2:8,:], traj.states[9:15,:]
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


