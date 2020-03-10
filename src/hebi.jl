using MuJoCo.MJCore: libmujoco, mjtNum, mjtByte, mjModel, mjData, Model, Data, MJVec
function mymj_ray(
    m::Model,
    d::Data,
    pnt::MJVec{mjtNum},
    vec::MJVec{mjtNum},
    geomgroup::Union{MJVec{mjtByte}, Ptr{Nothing}},
    flg_static::mjtByte,
    bodyexclude::Integer,
    geomid::MJVec{Cint},
)
    ccall(
        (:mj_ray, libmujoco),
        mjtNum,
        (
         Ptr{mjModel},
         Ptr{mjData},
         Ptr{mjtNum},
         Ptr{mjtNum},
         Ptr{mjtByte},
         mjtByte,
         Cint,
         Ptr{Cint},
        ),
        m,
        d,
        pnt,
        vec,
        geomgroup,
        flg_static,
        bodyexclude,
        geomid,
    )
end


using LyceumMuJoCo: fastreset_nofwd!

struct HebiPickup{S<:MJSim,A,O} <: AbstractMuJoCoEnvironment
    sim::S
    asp::A
    osp::O
    goal::SVector{3, Float64}
    ctrl::Vector
    function HebiPickup(sim::MJSim)
        goal = SA_F64[0.6, 0.03, 0.01]
        m = sim.m
        osp = MultiShape(qpos = VectorShape(Float64, m.nq), # no object
                         qvel = VectorShape(Float64, m.nv),
                         eff  = VectorShape(Float64, m.nv),

                         obj = VectorShape(Float64, 3),
                         goal = VectorShape(Float64, 3),

                         chopcenterpos = VectorShape(Float64, 3),
                         choprot = VectorShape(Float64, 3),

                         d_chopcenterpos2obj  = ScalarShape(Float64),
                         d_obj2goal = ScalarShape(Float64),
                         d_rot = ScalarShape(Float64),
                         d_choptips = ScalarShape(Float64),
                         iscentered = ScalarShape(Float64),
                         ispickedup = ScalarShape(Float64),
                        )
        asp = VectorShape(Float64, 7)

        #new{typeof(sim), typeof(asp), typeof(osp)}(sim, asp, osp, goal, zeros(eltype(sim.d.ctrl), size(sim.d.ctrl)...))
        new{typeof(sim), typeof(asp), typeof(osp)}(sim, asp, osp, goal, zeros(asp))
    end
end

function iscentered(e::HebiPickup)
    @unpack m, d, mn, dn = e.sim
    dx = dn.site_xpos[:, :chop] - dn.site_xpos[:, :stick]
    dx = dx / norm(dx)
    #pnt = dn.site_xpos[:, :stick] + dx * 0.003
    pnt = dn.site_xpos[:, :stick] + dx * 0.004
    geomid = Cint[-1]
    dist = mymj_ray(m, d, pnt, dx, C_NULL, UInt8(0), -1, geomid)
    first(geomid) == 22
end

function chopstickcenter(e::HebiPickup)
    @unpack m, d, mn, dn = e.sim
    dx = dn.site_xpos[:, :chop] - dn.site_xpos[:, :stick]
    dn.site_xpos[:, :stick] + dx / 2
end

function iscontacting(e::HebiPickup)
    # stick=19, chop=20, block=22
    stick = false
    chop = false
    for c = e.sim.d.contact
        stick |= (c.geom1 == 19 && c.geom2 == 22 || c.geom1 == 22 && c.geom2 == 19)
        chop |= (c.geom1 == 20 && c.geom2 == 22 || c.geom1 == 22 && c.geom2 == 20)
        #(c.geom1 == 19 || c.geom1 == 20) && c.geom2 == 22 && return true
        #(c.geom2 == 19 || c.geom2 == 20) && c.geom1 == 22 && return true
    end
    return stick && chop
end

function ispickedup(e::HebiPickup)
    for c = e.sim.d.contact
        c.geom1 == 22 && c.geom2 == 0 && return false
        c.geom1 == 0 && c.geom2 == 22 && return false
    end
    return iscontacting(e)
end

function tconstruct(::Type{HebiPickup}, n::Integer)
    modelpath = joinpath(@__DIR__, "hebi.xml")
    return Tuple(HebiPickup(s) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip = 20))
end

HebiPickup() = first(tconstruct(HebiPickup, 1))

@inline LyceumMuJoCo.getsim(env::HebiPickup) = env.sim
@inline LyceumMuJoCo.obsspace(env::HebiPickup) = env.osp


@propagate_inbounds function LyceumMuJoCo.reset!(env::HebiPickup)
    @unpack dn = env.sim
    fastreset_nofwd!(env.sim)
    #dn.qpos[Symbol("HEBI/shoulder/X8_16")] = 8pi/16
    #dn.qpos[Symbol("HEBI/elbow/X8_9")] = 11pi/16
    #dn.qpos[Symbol("HEBI/wrist1/X5_1")] = 3pi/16

    dn.qpos[Symbol("HEBI/shoulder/X8_16")] = 8pi/16
    dn.qpos[Symbol("HEBI/elbow/X8_9")] = 9pi/16
    dn.qpos[Symbol("HEBI/wrist1/X5_1")] = 1pi/16
    forward!(env.sim)
    env
end

@propagate_inbounds function LyceumMuJoCo.randreset!(rng::Random.AbstractRNG, env::HebiPickup)
    fastreset_nofwd!(env.sim)
    #keypos_position(env.sim)
    d = env.sim.d
    d.qpos[end-2] = rand(rng, Uniform(0.4, 0.7))
    d.qpos[end-1] = rand(rng, Uniform(-0.3, 0.3))
    forward!(env.sim)
    env
end

@inbounds _splat7(x::AbstractVector, i=0) = SA_F64[x[i+1], x[i+2], x[i+3], x[i+4], x[i+5], x[i+6], x[i+7]]
@inbounds _splat7(x::AbstractMatrix, i) = SA_F64[x[i,1], x[i,2], x[i,3], x[i,4], x[i,5], x[i,6], x[i,7]]

LyceumBase.actionspace(env::HebiPickup) = env.asp
@propagate_inbounds function LyceumMuJoCo.setaction!(env::HebiPickup, a)
    @unpack m, d, mn, dn = env.sim
    env.ctrl .= a


    efc, chop = a[1:6], a[7]
    jacpT = zeros(Int(env.sim.m.nv), 3)
    jacrT = zeros(Int(env.sim.m.nv), 3)
    MuJoCo.MJCore.mj_jacSite(m, d, vec(jacpT), vec(jacrT), 0) # not 1-based indexing??
    jac = vcat(jacpT', jacrT')[:, 1:6]

    dtheta = zeros(6)
    sdls_jacctrl!(dtheta, jac, efc, lammax = 75)
    #dls_jacctrl!(dtheta, jac, efc, lambda = 0.1)
    #velgain = [40, 30, 30, 30, 30, 30]
    #clamp!(dtheta, -0.25, 0.25)
    #dtheta = velgain .* dtheta #.+ -2.5 * d.qvel[1:6]

    dtheta[1:3] .= 1 * (dtheta[1:3] - d.qvel[1:3]) - 0.005 * d.qacc[1:3]
    dtheta[4:end] .= 10 * (dtheta[4:end] - d.qvel[4:6]) - 0.005 * d.qacc[4:6]

    chop = 100 * (chop - dn.qvel[Symbol("HEBI/chopstick/X5_1")])
    ctrl = [dtheta..., chop]

    clamp!(ctrl, -1.0, 1.0)

    a = ctrl

    maxtorque = _splat7(m.actuator_biasprm, 1)
    speed_24v = _splat7(m.actuator_biasprm, 2)
    qvel = _splat7(d.qvel)
    maxT = (maxtorque ./ speed_24v) .* abs.(qvel) .+ maxtorque

    d.ctrl .= a .* maxT

    env
end

LyceumMuJoCo.getaction!(a, env::HebiPickup) = copyto!(a, env.ctrl)

@inline _sitedist(s1, s2, dmin) = min(euclidean(s1, s2), dmin)
@propagate_inbounds function LyceumMuJoCo.getobs!(obs, env::HebiPickup)
    @unpack m, d, mn, dn = env.sim
    @unpack qpos, qvel = d

    obj = dn.geom_xpos[:, :obj]
    goal = dn.site_xpos[:, :goal]

    chopcenterpos = chopstickcenter(env)

    wRc = RotMatrix(dn.site_xmat[:, :chop]...)'
    wRf = RotXYZ(-pi/2, 0, 0) # flat chopsticks
    wRcf = wRf * transpose(wRc)

    @uviews obs @inbounds begin
        o = obsspace(env)(obs)
        o.qpos .= qpos
        o.qvel .= qvel
        o.eff  .= d.qfrc_actuator

        o.obj  .= obj
        o.goal  .= goal

        o.chopcenterpos .= chopcenterpos
        o.choprot .= rotation_axis(wRc) * rotation_angle(wRc)

        o.d_chopcenterpos2obj  = euclidean(chopcenterpos, obj)
        o.d_obj2goal = euclidean(goal, obj)
        o.d_rot = norm(rotation_axis(wRcf) * rotation_angle(wRcf))
        o.d_choptips = euclidean(dn.site_xpos[:, :chop], dn.site_xpos[:, :stick])
        o.iscentered = iscentered(env) ? 1.0 : 0.0
        o.ispickedup = ispickedup(env) ? 1.0 : 0.0
    end

    obs
end

@propagate_inbounds function LyceumMuJoCo.step!(env::HebiPickup)
    m, d = env.sim.m, env.sim.d

    #display(e.sim.d.qpos)
    # approximately calculated velocity limits from data; spec sheet isn't accurate
    vellimits = SVector{7, Float64}(3.596, 1.934, 3.596, 10.79, 10.79, 10.79, 10.79)

    for i=1:m.nu # only limit the hebi actuators, nothing else
        d.qvel[i] = clamp(d.qvel[i], -vellimits[i], vellimits[i])
    end

    step!(env.sim)
end

@propagate_inbounds function LyceumMuJoCo.getreward(state, action, obs, env::HebiPickup)
    @unpack m, d, mn, dn = env.sim
    osh = obsspace(env)(obs)

    reward = 0.0
    reward -= 1 * osh.d_chopcenterpos2obj

    dclosed, dopen = 0.005, 0.03

    if osh.ispickedup == 1.0
        reward += 1
        reward -= 1 * osh.d_obj2goal
        reward -= 0.1 * sqrt(osh.d_obj2goal)
        reward -= 5 * sqrt(osh.d_chopcenterpos2obj)
    elseif osh.d_chopcenterpos2obj < 0.03
        # start clamping
        reward += 0.03
        reward -= 0.1 * abs(osh.d_choptips - dclosed)
    else
        reward -= 0.1 * abs(osh.d_choptips - dopen)
    end

    return reward
end

LyceumMuJoCo.geteval(state, action, obs, env::HebiPickup) = 0

####
#### Util
####

## This function sets to the key_qpos but also sets the controls assuming it's position controllers
#function keypos_position(sim)
#    key_qpos = sim.m.key_qpos ## TODO OBJECT ## TODO OBJECT
#    @uviews key_qpos @inbounds sim.d.qpos .= view(key_qpos,:,1) # noalloc
#    sim.d.ctrl[2] = 1.0   # when using position control
#    sim.d.ctrl[3] = 1.0
#    sim.d.ctrl[7] = -0.21
#end