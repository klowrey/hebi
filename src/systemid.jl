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

function sysidFromFile(s, test::HebiPickup; optm=:NM, batch=nothing)
    dt, pos, vel, act = parse_csv_recordings(s)
    batch == nothing && (batch = 1:size(dt))
    return sysid(vcat(pos, vel), test, act; optm = optm, batch=batch)
end

function parse_csv_recordings(s="example.csv", nv=7)
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

function getHebiModelVars()
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
    setparams! = function (env::HebiPickup, p::AbstractVector)
        mp = modelvars(p)
        m = env.sim.m

        m.dof_damping[1]      = mp.x8_9_damping
        m.dof_damping[2]      = mp.x8_16_damping
        m.dof_damping[3]      = mp.x8_9_damping
        m.dof_damping[4:7]   .= mp.x5_1_damping

        m.dof_armature[1]     = mp.x8_9_armature
        m.dof_armature[2]     = mp.x8_16_armature
        m.dof_armature[3]     = mp.x8_9_armature
        m.dof_armature[4:7]  .= mp.x5_1_armature

        m.dof_frictionloss[1] = mp.x8_9_frictionloss
        m.dof_frictionloss[2] = mp.x8_16_frictionloss
        m.dof_frictionloss[3] = mp.x8_9_frictionloss
        m.dof_frictionloss[4:7].= mp.x5_1_frictionloss

        m.actuator_gear[1,1]    = mp.x8_9_gear
        m.actuator_gear[1,2]    = mp.x8_16_gear
        m.actuator_gear[1,3]    = mp.x8_9_gear
        m.actuator_gear[1,4:7] .= mp.x5_1_gear

        env
    end
    getparams! = function (p::AbstractVector, env::HebiPickup)
        mp = modelvars(p)
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

    return modelvars, getparams!, setparams!
end

function sysid(refstate, test::HebiPickup, ctrls; optm=:NM,
               batch=1:size(refstate,2),
               lower=0.001, upper=100.0)
    modelvars, getparams!, setparams! = getHebiModelVars()

    N = length(modelvars)
    osp = obsspace(test)

    nq = test.sim.m.nq
    nv = test.sim.m.nv

    function opt(P, env=test, Y=refstate, batch=batch)
        setparams!(env, P)

        reset!(env)
        env.sim.d.qpos .= Y[1:nq,  batch.start]
        env.sim.d.qvel .= Y[nq+1:nq+nv, batch.start]
        env.sim.d.ctrl .= ctrls[:,batch.start]
        forward!(env.sim)
        roll = _rollout(env, ctrls[:,batch])

        pidx = 1:nq
        vidx = nq .+ pidx
        return mse(Y[pidx,batch], roll.obses[pidx,:]) + mse(Y[vidx,batch], roll.obses[vidx,:])
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
            cache[i] = opt(Peps[i], tests[tid])
        end
        storage .= (cache[1:N] .- cache[end]) ./ ep
    end

    initP = allocate(modelvars)
    getparams!(initP, test)

    isa(lower, Number) && (lower = fill(lower, N)) # approx upper and lower bounds? can specialize more...
    isa(upper, Number) && (upper = fill(upper, N))
    initP .= clamp.(initP, lower, upper)

    options = Optim.Options(allow_f_increases=false,
                            show_trace=true, show_every=10,
                            iterations=40000, time_limit=60*60)
    if optm == :NM
        #result = optimize(opt, initP, NelderMead(; initial_simplex=MySimplexer{T}(xmax, 0.0)), options)
        result = optimize(p->opt(p, test), lower, upper, initP,
                          Fminbox(NelderMead()), options) # probably need custom simplex initializer
        setparams!(test, result.minimizer)#, modelvars)
    elseif optm == :LBFGS
        #result = optimize(opt, lower, upper, initP, Fminbox(LBFGS()), options)
        result = optimize(p->opt(p, test), optgrad!,
                          lower, upper, initP,
                          Fminbox(LBFGS(alphaguess = LineSearches.InitialQuadratic(),
                                        linesearch = LineSearches.BackTracking())),
                          options)
        setparams!(test, result.minimizer)#, modelvars)
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

function getHebiSinCtrl(env::HebiPickup; T=2000)

    function hebiSinFn(env::HebiPickup)
        m, d = env.sim.m, env.sim.d
        t = 2 * d.time
        d.ctrl[1] = sin(t)
        d.ctrl[2] = 1.0 + 0.25 * sin(t)
        d.ctrl[3] = 0.6 + 0.25 * sin(2t)
        d.ctrl[4] = sin(t)
        d.ctrl[5] = 0.1 * sin(2t)
        d.ctrl[6] = sin(t)
        d.ctrl[7] = -0.21 + 0.05 * sin(t)
    end

    reset!(env)
    actions = zeros(length(actionspace(env)), T)
    for t=1:T
        hebiSinFn(env) # set controls
        getaction!(view(actions, :, t), env) # save actions
        step!(env)
    end
    reset!(env)
    actions
end

function testSysidOnevar(ref::HebiPickup, test::HebiPickup; ctrls=nothing, optm=:NM)
    # Optionally only allow one param
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
    modelvars, setparams!, getparams! = getHebiModelVars()

    initP = allocate(modelvars)
    getparams!(initP, ref)
    initP .+= 0.01
    clamp!(initP, 0.001, 4.9)
    println("Initial params: ", initP)

    reset!(ref)
    ctrls == nothing && (ctrls = getHebiSinCtrl(ref))
    traj = _rollout(ref, ctrls)

    resetset!(test, ref)
    test.sim.m.dof_damping[2] = 4.9
    test.sim.m.dof_damping[3] = 4.9
    getparams!(initP, test)
    println("Disturbed params: ", initP)

    result = sysid(traj.obses, test, ctrls; optm=optm, lower=0.0001, upper=5.0)

    getparams!(initP, ref)
    println("Diff from reference model:")
    println(round.(initP .- result.minimizer, digits=3))

    return result
end

function testSysid(ref::HebiPickup, test::HebiPickup; ctrls=nothing, optm=:NM)
    modelvars, getparams!, setparams! = getHebiModelVars()

    initP = allocate(modelvars)
    getparams!(initP, ref)
    clamp!(initP, 0.001, 4.9)
    println("Initial params: ", initP)

    reset!(ref)
    ctrls == nothing && (ctrls = getHebiSinCtrl(ref))
    traj = _rollout(ref, ctrls)

    randset!(test, ref)
    getparams!(initP, test)
    clamp!(initP, 0.001, 4.9)
    setparams!(test, initP)
    println("Disturbed params: ", initP)

    result = sysid(traj.obses, test, ctrls; optm=optm, lower=0.0001, upper=5.0)

    getparams!(initP, ref)
    println("Diff from reference model:")
    println(round.(initP .- result.minimizer, digits=3))

    return result
end
