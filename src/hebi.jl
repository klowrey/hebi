
using LyceumMuJoCo: fastreset_nofwd!

struct HebiPickup{S<:MJSim, O} <: AbstractMuJoCoEnvironment
    sim::S
    osp::O
    goal::SVector{3, Float64}
    function HebiPickup(sim::MJSim)
        goal = SA_F64[0.5, 0.0, 0.2]
        m = sim.m
        #Josp = MultiShape(qpos = VectorShape(Float64, m.nq-7),  ## TODO OBJECT# no object
        osp = MultiShape(qpos = VectorShape(Float64, m.nq), # no object
                         qvel = VectorShape(Float64, m.nv),
                         eff  = VectorShape(Float64, m.nv),
                         obj  = VectorShape(Float64, 3),
                         cpsk  = VectorShape(Float64, 3), # chopstick site
                         d_obj  = ScalarShape(Float64), # distance
                         d_goal = ScalarShape(Float64),
                         cs2obj  = VectorShape(Float64, 3), # chopsticks to object
                         obj2goal = VectorShape(Float64, 3), # object to goal
                         dp = VectorShape(Float64, 3),
                         dr = VectorShape(Float64, 3)
                        )

        #my_warning_cb(msg::Cstring) = (@warn unsafe_string(msg); nothing)
        #warncb = @cfunction(hebicontroller, mjModel, mjData, (model, data))
        #MJCore.CGlobals.mjcb_control = warncb

        new{typeof(sim), typeof(osp)}(sim, osp, goal)
    end
end

#function hebicontroller(m, d)
#
#end

function tconstruct(::Type{HebiPickup}, n::Integer)
    modelpath = joinpath(@__DIR__, "hebi.xml")
    return Tuple(HebiPickup(s) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip = 5))
end
HebiPickup() = first(tconstruct(HebiPickup, 1))

@inline LyceumMuJoCo.getsim(env::HebiPickup) = env.sim
@inline LyceumMuJoCo.obsspace(env::HebiPickup) = env.osp

# This function sets to the key_qpos but also sets the controls assuming it's position controllers
function keypos_position(sim)
    key_qpos = sim.m.key_qpos ## TODO OBJECT ## TODO OBJECT
    @uviews key_qpos @inbounds sim.d.qpos .= view(key_qpos,:,1) # noalloc
    sim.d.ctrl[2] = 1.0   # when using position control
    sim.d.ctrl[3] = 1.0
    sim.d.ctrl[7] = -0.21
end

@propagate_inbounds function LyceumMuJoCo.reset!(env::HebiPickup)
    fastreset_nofwd!(env.sim)
    env.sim.dn.qpos[Symbol("HEBI/shoulder/X8_16")] = pi/4
    env.sim.dn.qpos[Symbol("HEBI/elbow/X8_9")] = pi/4
    ##keypos_position(env.sim) # when using position control
    forward!(env.sim)
    env
end

@propagate_inbounds function LyceumMuJoCo.randreset!(rng::Random.AbstractRNG, env::HebiPickup)
    fastreset_nofwd!(env.sim)
    #keypos_position(env.sim)
    d = env.sim.d
    d.qpos[end-2] = rand(rng, Uniform(0.4, 0.7))
    d.qpos[end-1] = rand(rng, Uniform(-0.3, 0.3))
    d.qpos[end-0] = 0.02
    forward!(env.sim)
    env
end

@inbounds _splat7(x::AbstractVector, i=0) = SA_F64[x[i+1], x[i+2], x[i+3], x[i+4], x[i+5], x[i+6], x[i+7]]
@inbounds _splat7(x::AbstractMatrix, i) = SA_F64[x[i,1], x[i,2], x[i,3], x[i,4], x[i,5], x[i,6], x[i,7]]

@propagate_inbounds function LyceumMuJoCo.setaction!(env::HebiPickup, a)
    m, d = env.sim.m, env.sim.d

    maxtorque = _splat7(m.actuator_biasprm, 1)
    speed_24v = _splat7(m.actuator_biasprm, 2)

    qvel = _splat7(d.qvel)
    #ctrl = SVector{7, Float64}(a)

    #scale = (maxtorque ./ speed_24v) .* qvel
    #maxT = scale .+ maxtorque # where we are on the speed-torque curve; y=mx+b
    #d.ctrl .= ctrl .* maxT .+ scale

    maxT = (maxtorque ./ speed_24v) .* abs.(qvel) .+ maxtorque
    d.ctrl .= a .* maxT #.+ (maxtorque ./ speed_24v) .* qvel

    env
end

@inline _sitedist(s1, s2, dmin) = min(euclidean(s1, s2), dmin)
@propagate_inbounds function LyceumMuJoCo.getobs!(obs, env::HebiPickup)
    m, d = env.sim.m, env.sim.d

    obj = m.ngeom

    qpos = d.qpos
    qvel = d.qvel

    dmin = 0.5
    _obj  = SPoint3D(d.geom_xpos, obj)
    _cpsk = SPoint3D(d.site_xpos, 1)


    dn = env.sim.dn
    dp = dn.xpos[:, :target] - dn.site_xpos[:, :chopstick]
    r_target = RotMatrix(dn.site_xmat[:, :target]...)
    r_chopstick = RotMatrix(dn.site_xmat[:, :chopstick]...)

    @uviews obs @inbounds begin
        o = obsspace(env)(obs)
        #o.qpos .= view(qpos, 1:(m.nq-7)) ## TODO OBJECT
        o.qpos .= view(qpos, 1:(m.nq))
        o.qvel .= view(qvel, 1:m.nv)
        o.eff  .= d.qfrc_actuator
        o.obj  .= _obj
        o.cpsk .= _cpsk
        o.d_obj  = _sitedist(_obj, _cpsk, dmin)
        o.d_goal = _sitedist(_obj, env.goal, dmin)
        o.cs2obj .= _obj - _cpsk
        o.obj2goal .= _obj - env.goal
        o.dp .= dp
    end
    obs
end

@propagate_inbounds function LyceumMuJoCo.step!(env::HebiPickup)
    m, d = env.sim.m, env.sim.d

    # approximately calculated velocity limits from data; spec sheet isn't accurate
    vellimits = SVector{7, Float64}(3.596, 1.934, 3.596, 10.79, 10.79, 10.79, 10.79)

    for i=1:m.nu # only limit the hebi actuators, nothing else
        d.qvel[i] = clamp(d.qvel[i], -vellimits[i], vellimits[i])
    end

    step!(env.sim)
end

@propagate_inbounds function LyceumMuJoCo.getreward(state, action, obs, env::HebiPickup)
    #o = obsspace(env)(obs)

    #_cs2obj = o.d_obj / 0.5
    #_obj2goal = o.d_goal / 0.5

    #reward = -_cs2obj
    #if _cs2obj < 0.006
    #    reward = 2.0 - 2 * _obj2goal
    #end
    #reward
    dn = env.sim.dn
    dp = dn.xpos[:, :target] - dn.site_xpos[:, :chopstick]
    r_target = RotMatrix(dn.site_xmat[:, :target]...)
    r_chopstick = RotMatrix(dn.site_xmat[:, :chopstick]...)

    return 0
    #return -norm(dp) #- norm(dr)
end

@propagate_inbounds function LyceumMuJoCo.geteval(state, action, obs, env::HebiPickup)
    return 0.0
end

