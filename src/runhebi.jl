using LinearAlgebra
using LyceumAI, LyceumBase, LyceumMuJoCo, LyceumMuJoCoViz, Shapes
using LyceumBase.Tools
using UniversalLogger
using Base: @propagate_inbounds
using Random
import LyceumBase: tconstruct, _rollout
using Statistics, StaticArrays, UnsafeArrays
using Base.Iterators
using Distributions, Distances
using MuJoCo

using CSV

using Optim
using LineSearches

using BlackBoxOptim
using Distributed
using SharedArrays

include("hebi.jl")
include("util.jl")
include("controls.jl")




function sysidfromfile(s, test::HebiPickup; optm=:NM, batch=nothing)
    dt, pos, vel, act = datafile(s)

    if batch == nothing
        batch = 1:size(pos,2)
    end
    return filesysid(vcat(pos, vel), test, act; optm = optm, batch=batch)
end

function datafile(s="example.csv", nv=7, skip=4)
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
    if skip > 1
        dt = dt[2:skip:end]
        println("Average dt, processed: ", mean(diff(dt)))
        println("num datapoints: ", size(dt, 1))
        return dt, pos[:,2:skip:end], vel[:,2:skip:end], act[:,2:skip:end], eff[:,2:skip:end]
    else
        println("Average dt, raw: ", mean(diff(dt)))
        return dt, pos, vel, act, eff 
    end
end

function gethebivars()
    return MultiShape(x8_damping         = ScalarShape(Float64),
                      x8_armature        = ScalarShape(Float64),
                      x8_frictionloss    = ScalarShape(Float64),
                      x8_maxT            = ScalarShape(Float64),
                      x8_ramp            = ScalarShape(Float64),

                      x8_16_damping      = ScalarShape(Float64),
                      x8_16_armature     = ScalarShape(Float64),
                      x8_16_frictionloss = ScalarShape(Float64),
                      x8_16_maxT         = ScalarShape(Float64),
                      x8_16_ramp         = ScalarShape(Float64),

                      x5_damping         = ScalarShape(Float64),
                      x5_armature        = ScalarShape(Float64),
                      x5_frictionloss    = ScalarShape(Float64),
                      x5_maxT            = ScalarShape(Float64),
                      x5_ramp            = ScalarShape(Float64)
                     )
end

function setparams!(env, p::AbstractVector, varspace)
    mp = varspace(p)
    m = env.sim.m

    m.dof_damping[[1,3]]        .= mp.x8_damping
    m.dof_armature[[1,3]]       .= mp.x8_armature
    m.dof_frictionloss[[1,3]]   .= mp.x8_frictionloss
    m.actuator_biasprm[1,[1,3]] .= mp.x8_maxT
    m.actuator_biasprm[2,[1,3]] .= mp.x8_ramp

    m.dof_damping[2]             = mp.x8_16_damping
    m.dof_armature[2]            = mp.x8_16_armature
    m.dof_frictionloss[2]        = mp.x8_16_frictionloss
    m.actuator_biasprm[1,2]      = mp.x8_16_maxT
    m.actuator_biasprm[2,2]      = mp.x8_16_ramp

    m.dof_damping[4:7]          .= mp.x5_damping
    m.dof_armature[4:7]         .= mp.x5_armature
    m.dof_frictionloss[4:7]     .= mp.x5_frictionloss
    m.actuator_biasprm[1,4:7]   .= mp.x5_maxT
    m.actuator_biasprm[2,4:7]   .= mp.x5_ramp

    env
end
function getparams!(p::AbstractVector, env, varspace)
    mp = varspace(p)
    m = env.sim.m

    mp.x8_damping         = m.dof_damping[1]
    mp.x8_armature        = m.dof_armature[1]
    mp.x8_frictionloss    = m.dof_frictionloss[1]
    mp.x8_maxT            = m.actuator_biasprm[1,1]
    mp.x8_ramp            = m.actuator_biasprm[2,1]

    mp.x8_16_damping      = m.dof_damping[2]
    mp.x8_16_armature     = m.dof_armature[2]
    mp.x8_16_frictionloss = m.dof_frictionloss[2]
    mp.x8_16_maxT         = m.actuator_biasprm[1,2]
    mp.x8_16_ramp         = m.actuator_biasprm[2,2]

    mp.x5_damping         = m.dof_damping[4]
    mp.x5_armature        = m.dof_armature[4]
    mp.x5_frictionloss    = m.dof_frictionloss[4]
    mp.x5_maxT            = m.actuator_biasprm[1,4]
    mp.x5_ramp            = m.actuator_biasprm[2,4]

    env
end

function filesysid(refstate, test::HebiPickup, ctrls, modelvars=gethebivars();
                   optm=:NM, batch = 1:size(refstate, 2), horizon=batch[end])
    N = length(modelvars)
    osp = obsspace(test)
    s = size(refstate, 1) # length(osp.qpos) + length(osp.qvel) #- 6 # ignore the object's data for now
    println(s)

    #batch = 1500:8000 #1000:6000
    nq = test.sim.m.nq
    nv = test.sim.m.nv

    function opt(P, env=test, Y=refstate, batch=batch)
        setparams!(env, P, modelvars)

        reset!(env)
        env.sim.d.qpos .= view(Y, 1:7,  batch[1])
        env.sim.d.qvel .= view(Y, 8:14, batch[1])
        env.sim.d.ctrl .= view(ctrls,:, batch[1])
        forward!(env.sim)
        roll = _sysidrollout(env, view(ctrls, :,batch))

        #return mse(Y[:,batch], roll.obses[1:s,:]) # from LyceumAI
        idx = 1:7 #4:6
        vid = nq .+ idx
        return mse(Y[idx,batch], roll.obses[idx,:]) + 5*mse(Y[vid,batch], roll.obses[vid,:])
    end

    tests = [ HebiPickup() for _=1:Threads.nthreads() ] # independent models for parallel eval
    shared = tconstruct(HebiPickup, Threads.nthreads()) # shared model for parallel rollouts
    function t_opt(P, env=test, Y=refstate; batch=batch, horizon=horizon, nthreads=Threads.nthreads())

        batches = collect(partition(batch, horizon))
        length(batches[end]) < length(batches[1]) && pop!(batches)
        nbatch = length(batches)

        threadrange = collect(partition(1:nbatch, max(1,div(nbatch, nthreads))))
        
        cost = zeros(nthreads)
        @sync for i=1:min(length(threadrange), nthreads)
            Threads.@spawn begin
                tid = Threads.threadid() 
                for b in batches[threadrange[i]]
                    cost[i] += opt(P, tests[tid], Y, b)
                end
                cost[i] /= length(threadrange[i])
            end
        end
        #Threads.@threads for b in batches
        #for b in batches
        #    tid = Threads.threadid()
        #    if length(b) == horizon # only get full batches
        #        cost[tid] += opt(P, tests[tid], Y, b)
        #    end
        #end
        return sum(cost) / length(threadrange) 
    end

    Peps = [ zeros(N) for i=1:N+1 ]
    function optgrad!(storage, P, ep=1e-3)
        for i=1:N
            Peps[i] .= P
            Peps[i][i] += ep 
        end
        Peps[end] .= P

        cache = zeros(N+1)
        Threads.@threads for i=1:(N+1)
            tid = Threads.threadid() 
            #cache[i] = t_opt(Peps[i], tests[tid], nthreads=1)
            cache[i] = opt(Peps[i], tests[tid])
        end
        storage .= (cache[1:N] .- cache[end]) ./ ep
    end

    initP = allocate(modelvars)
    getparams!(initP, test, modelvars)
    println(initP)
    println(N)
    hi = 2.0
    lo = 0.01
    #lower = fill(lo, N) # approx upper and lower bounds? can specialize more...
    lower = initP .* 0.5
    upper = initP .* 1.1 .+ 2.0
    println("upper:")
    display(upper)

    options = Optim.Options(allow_f_increases=false,
                            #outer_iterations=20, # fminbox iterations
                            show_trace=true, show_every=10, x_tol=1e-3, g_tol=1e-5,
                            iterations=40000, time_limit=60*160)
    if optm == :NM
        #result = optimize(opt, initP, NelderMead(; initial_simplex=MySimplexer{T}(xmax, 0.0)), options)
        result = optimize(t_opt, lower, upper, initP, Fminbox(NelderMead()), options) # probably need custom simplex initializer
        setparams!(test, result.minimizer, modelvars)
    elseif optm == :LBFGS
        #result = optimize(opt, lower, upper, initP, Fminbox(LBFGS()), options)
        #result = optimize(t_opt, optgrad!,
        result = optimize(opt, optgrad!,
                          lower, upper, initP,
                          Fminbox(LBFGS(alphaguess = LineSearches.InitialQuadratic(),
                                        linesearch = LineSearches.BackTracking())),
                          options)
        setparams!(test, result.minimizer, modelvars)
    elseif optm == :PS
        @warn "optimizer $optm not configured yet."
        result = optimize(opt, initP,
                          ParticleSwarm(lower=lower, upper=upper, n_particles=18), options) # doesn't work?
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
        roll = _sysidrollout(env, ctrls)

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


