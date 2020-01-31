using LinearAlgebra
using LyceumAI, LyceumBase, LyceumMuJoCo, LyceumMuJoCoViz, Shapes
using LyceumBase.Tools
using UniversalLogger
using Base: @propagate_inbounds
using Random
using CSV
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

function sysidfromfile(s, test::HebiPickup; optm=:NM, batch=nothing)
    dt, pos, vel, act = datafile(s)
    batch == nothing && (batch = 1:size(dt)) # was size(pos, 2)
    return filesysid(vcat(pos, vel), test, act; optm = optm, batch=batch)
end

function datafile(s="example.csv", nv=7)
    d = CSV.read(s)
    r, c = size(d)
    println("Row #: $(r)\n", names(d))

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

    dt = (d.timestamp_us .- d.timestamp_us[1]) * 1e-6

    # to currently match the 0.004 DT model (dt = 0.002 with skip = 2)
    # return dt[2:4:end], pos[:,2:4:end], vel[:,2:4:end], act[:,2:4:end]
    return dt, pos, vel, act, eff
end

function filesysid(refstate, test::HebiPickup, ctrls; optm=:NM, batch=1:size(refstate,2))
    modelvars = MultiShape(x8_9_damping         = ScalarShape(Float64),
                           x8_9_armature        = ScalarShape(Float64),
                           x8_9_frictionloss    = ScalarShape(Float64),
                           x8_9_gear            = ScalarShape(Float64),
                           x8_16_damping        = ScalarShape(Float64),
                           x8_16_armature       = ScalarShape(Float64),
                           x8_16_frictionloss   = ScalarShape(Float64),
                           x8_16_gear           = ScalarShape(Float64),
                           x5_1_damping         = ScalarShape(Float64),
                           x5_1_armature        = ScalarShape(Float64),
                           x5_1_frictionloss    = ScalarShape(Float64),
                           x5_1_gear            = ScalarShape(Float64),
                          )
    N = length(modelvars)
    function setparams!(env, p::AbstractVector, varspace)
        mp = varspace(p)
        m = env.sim.m

        m.dof_damping[1]      = mp.x8_9_damping
        m.dof_damping[2]      = mp.x8_16_damping
        m.dof_damping[3]      = mp.x8_9_damping
        m.dof_damping[4:7]     = mp.x5_1_damping

        m.dof_armature[1]     = mp.x8_9_armature
        m.dof_armature[2]     = mp.x8_16_armature
        m.dof_armature[3]     = mp.x8_9_armature
        m.dof_armature[4:7]    = mp.x8_9_armature

        m.dof_frictionloss[1] = mp.x8_9_frictionloss
        m.dof_frictionloss[2] = mp.x8_16_frictionloss
        m.dof_frictionloss[3] = mp.x8_9_frictionloss
        m.dof_frictionloss[4:7] = mp.x8_9_frictionloss

        m.actuator_gear[1,1]    = mp.x8_9_gear
        m.actuator_gear[1,2]    = mp.x8_16_gear
        m.actuator_gear[1,3]    = mp.x8_9_gear
        m.actuator_gear[1,4:7] .= mp.x5_1_gear

        env
    end
    function getparams!(p::AbstractVector, env, varspace)
        mp = varspace(p)
        m = env.sim.m

        mp.x8_9_damping         = m.dof_damping[1]
        mp.x8_9_armature        = m.dof_armature[1]
        mp.x8_9_frictionloss    = m.dof_frictionloss[1]
        mp.x8_9_gear            = m.actuator_gear[1,1]

        mp.x8_16_damping        = m.dof_damping[2]
        mp.x8_16_armature       = m.dof_armature[2]
        mp.x8_16_frictionloss   = m.dof_frictionloss[2]
        mp.x8_16_gear           = m.actuator_gear[1,2]

        mp.x5_1_damping        = m.dof_damping[4]
        mp.x5_1_armature       = m.dof_armature[4]
        mp.x5_1_frictionloss   = m.dof_frictionloss[4]
        mp.x5_1_gear           = m.actuator_gear[1,4]

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
    modelvars = MultiShape(x8_9_damping      = ScalarShape(Float64),
                           x5_1_damping      = ScalarShape(Float64),
                           x8_9_armature     = ScalarShape(Float64),
                           x5_1_armature     = ScalarShape(Float64),
                           x8_9_frictionloss = ScalarShape(Float64),
                           x5_1_frictionloss = ScalarShape(Float64))
    scalingfactor = [ 1000.0, 100.0, 10.0, 10.0, 1000.0, 10.0 ]
    N = length(modelvars)
    function setparams!(env, p::AbstractVector)
        mp = modelvars(p)
        env.sim.m.dof_damping[1:3]      .= scalingfactor[1] * mp.x8_9_damping
        env.sim.m.dof_damping[4:7]      .= scalingfactor[2] * mp.x5_1_damping
        env.sim.m.dof_armature[1:3]     .= scalingfactor[3] * mp.x8_9_armature
        env.sim.m.dof_armature[4:7]     .= scalingfactor[4] * mp.x5_1_armature
        env.sim.m.dof_frictionloss[1:3] .= scalingfactor[5] * mp.x8_9_frictionloss
        env.sim.m.dof_frictionloss[4:7] .= scalingfactor[6] * mp.x5_1_frictionloss
        env
    end
    function getparams!(p::AbstractVector, env)
        mp = modelvars(p)
        mp.x8_9_damping      = env.sim.m.dof_damping[1]      / scalingfactor[1]
        mp.x5_1_damping      = env.sim.m.dof_damping[4]      / scalingfactor[2]
        mp.x8_9_armature     = env.sim.m.dof_armature[1]     / scalingfactor[3]
        mp.x5_1_armature     = env.sim.m.dof_armature[4]     / scalingfactor[4]
        mp.x8_9_frictionloss = env.sim.m.dof_frictionloss[1] / scalingfactor[5]
        mp.x5_1_frictionloss = env.sim.m.dof_frictionloss[4] / scalingfactor[6]
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
