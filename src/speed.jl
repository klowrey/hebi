using MuJoCo
using MuJoCo.MJCore
using LyceumMuJoCo

@inline function mymj_step1(m, d)
    #mj_checkPos(m, d)
    #mj_checkVel(m, d)
    mj_fwdPosition(m, d)
    #mj_sensorPos(m, d)
    #mj_energyPos(m, d)
    mj_fwdVelocity(m, d)
    #mj_sensorVel(m, d)
    #mj_energyVel(m, d)
end

@inline function mymj_step2(m, d)
    mj_fwdActuation(m, d)
    mj_fwdAcceleration(m, d)
    mj_fwdConstraint(m, d)
    #mj_sensorAcc(m, d)
    #mj_checkAcc(m, d)
    mj_Euler(m, d)
end

function mymj_step(m, d)
    dt = ifelse(d.ncon > 0, 0.001, 0.004)
    #@set! m.opt.timestep = dt
    opt = m.opt
    m.opt = LyceumBase.SetfieldImpl.Setfield.@set opt.timestep = dt
    #@assert m.opt.timestep == dt
    mj_step1(m, d)
    mj_step2(m, d)
end

function test()
    e = HebiPickup();
    reset!(e)
    sim = e.sim
    setup = let sim=sim
        function ()
            reset!(e)
            dn = sim.dn
            dn.qpos[Symbol("HEBI/shoulder/X8_16")] = 8pi/16
            dn.qpos[Symbol("HEBI/elbow/X8_9")] = 11pi/16
            dn.qpos[Symbol("HEBI/wrist1/X5_1")] = 3pi/16
            forward!(sim)
            sim.m, sim.d
        end
    end
    @unpack m,d = sim
    #@btime mj_step(_m, _d) evals=10 samples=1000 setup=((_m, _d) = $setup())
    #@btime mymj_step(_m, _d) evals=10 samples=1 setup=((_m, _d) = $setup())
    x = Set{Int}()
    for i=1:10000
        mymj_step(sim.m, sim.d)
        push!(x, Int(sim.d.ncon))
    end
    x, e
end
