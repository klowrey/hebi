struct HEBIPickup{ActionMode,Sim<:MJSim,A,O} <: AbstractMuJoCoEnvironment
    sim::Sim
    asp::A
    osp::O
    lastaction::Vector{Float64}
    function HEBIPickup(sim::MJSim; action_mode::Symbol = :efc_vel)

        m = sim.m
        osp = MultiShape(
            qpos = VectorShape(Float64, m.nq),
            qvel = VectorShape(Float64, m.nv),
            eff = VectorShape(Float64, m.nv),
            obj = VectorShape(Float64, 3),
            goal = VectorShape(Float64, 3),
            chopcenterpos = VectorShape(Float64, 3),
            choprot = VectorShape(Float64, 3),
            d_chopcenterpos2obj = ScalarShape(Float64),
            d_obj2goal = ScalarShape(Float64),
            d_choptips = ScalarShape(Float64),
            iscentered = ScalarShape(Float64),
            ispickedup = ScalarShape(Float64),
        )

        if action_mode === :efc_vel
            asp = MultiShape(efc = VectorShape(Float64, 6), chop = ScalarShape(Float64))
        elseif action_mode === :joint_torque
            asp = actionspace(sim)
        else
            error("action_mode must be one of (:efc_vel, :joint_torque). Got: :$action_mode.")
        end

        new{action_mode,typeof(sim),typeof(asp),typeof(osp)}(sim, asp, osp, zeros(asp))
    end
end

HEBIPickup(args...; kwargs...) = first(tconstruct(HEBIPickup, 1, args...; kwargs...))

function LyceumMuJoCo.tconstruct(::Type{<:HEBIPickup}, n::Integer, args...; kwargs...)
    modelpath = joinpath(@__DIR__, "hebi.xml")
    return Tuple(
        HEBIPickup(s, args...; kwargs...) for s in tconstruct(MJSim, n, modelpath, skip = 20)
    )
end


function LyceumMuJoCo.reset!(env::HEBIPickup)
    @unpack dn = env.sim
    reset_nofwd!(env.sim)
    dn.qpos[Symbol("HEBI/shoulder/X8_16")] = 8pi / 16
    dn.qpos[Symbol("HEBI/elbow/X8_9")] = 9pi / 16
    dn.qpos[Symbol("HEBI/wrist1/X5_1")] = 1pi / 16
    forward!(env.sim)
    return env
end

function LyceumMuJoCo.randreset!(rng::Random.AbstractRNG, env::HEBIPickup)
    dn = env.sim.dn
    reset!(env)
    dn.qpos[:obj_x] += rand(rng, Uniform(-0.1, 0.1))
    dn.qpos[:obj_y] += rand(rng, Uniform(-0.2, 0.2))
    forward!(env.sim)
    return env
end


LyceumBase.actionspace(env::HEBIPickup) = env.asp

LyceumMuJoCo.getaction!(a, env::HEBIPickup) = copyto!(a, env.lastaction)

function LyceumMuJoCo.setaction!(env::HEBIPickup{:efc_vel}, a)
    @unpack m, d, mn, dn = env.sim
    env.lastaction .= a

    efc, chop = a[1:6], a[7]

    jacpT = zeros(Int(m.nv), 3)
    jacrT = zeros(Int(m.nv), 3)
    MuJoCo.MJCore.mj_jacSite(m, d, vec(jacpT), vec(jacrT), 0) # not 1-based indexing??
    jac = vcat(jacpT', jacrT')[:, 1:6]
    dtheta = zeros(6)
    sdls_jacctrl!(dtheta, jac, efc, lammax = 75)

    dtheta[1:3] .= 1 * (dtheta[1:3] - d.qvel[1:3]) - 0.005 * d.qacc[1:3]
    dtheta[4:end] .= 10 * (dtheta[4:end] - d.qvel[4:6]) - 0.005 * d.qacc[4:6]

    chop = 100 * (chop - dn.qvel[Symbol("HEBI/chopstick/X5_1")])
    ctrl = [dtheta..., chop]

    return _set_joint_torque!(env, ctrl)
end

function LyceumMuJoCo.setaction!(env::HEBIPickup{:joint_torque}, a)
    env.lastaction .= a
    _set_joint_torque!(env, a)
end

function _set_joint_torque!(env::HEBIPickup, torque)
    @unpack m, d = env.sim
    clamp!(torque, -1.0, 1.0)

    maxtorque = SVector(ntuple(i -> m.actuator_biasprm[1, i], Val(7)))
    speed_24v = SVector(ntuple(i -> m.actuator_biasprm[2, i], Val(7)))
    qvel = SVector(ntuple(i -> d.qvel[i], Val(7)))

    maxT = (maxtorque ./ speed_24v) .* abs.(qvel) .+ maxtorque

    d.ctrl .= torque .* maxT

    return env
end


@inline LyceumMuJoCo.obsspace(env::HEBIPickup) = env.osp

function LyceumMuJoCo.getobs!(obs, env::HEBIPickup)
    @unpack m, d, mn, dn = env.sim
    @unpack qpos, qvel = d

    obj = dn.geom_xpos[:, :obj]
    goal = dn.site_xpos[:, :goal]
    chopcenterpos = chopstickcenter(env)
    wRc = RotMatrix(dn.site_xmat[:, :chop]...)'

    @uviews obs @inbounds begin
        o = obsspace(env)(obs)
        o.qpos .= qpos
        o.qvel .= qvel
        o.eff .= d.qfrc_actuator

        o.obj .= obj
        o.goal .= goal

        o.chopcenterpos .= chopcenterpos
        o.choprot .= rotation_axis(wRc) * rotation_angle(wRc)

        o.d_chopcenterpos2obj = euclidean(chopcenterpos, obj)
        o.d_obj2goal = euclidean(goal, obj)
        o.d_choptips = euclidean(dn.site_xpos[:, :chop], dn.site_xpos[:, :stick])
        o.iscentered = iscentered(env) ? 1.0 : 0.0
        o.ispickedup = ispickedup(env) ? 1.0 : 0.0
    end

    return obs
end

function LyceumMuJoCo.step!(env::HEBIPickup)
    @unpack m, d = env.sim
    # approximately calculated velocity limits from data; spec sheet isn't accurate
    vel_limits = SVector{7,Float64}(3.596, 1.934, 3.596, 10.79, 10.79, 10.79, 10.79)
    for i = 1:m.nu # only limit the hebi actuators, nothing else
        d.qvel[i] = clamp(d.qvel[i], -vel_limits[i], vel_limits[i])
    end
    step!(env.sim)
    return env
end

function LyceumMuJoCo.getreward(state, action, obs, env::HEBIPickup)
    @unpack m, d, mn, dn = env.sim
    osh = obsspace(env)(obs)
    dclosed, dopen = 0.005, 0.03

    reward = 0.0
    reward -= 1 * osh.d_chopcenterpos2obj

    reward -= 1 * osh.d_obj2goal
    if osh.ispickedup == 1.0
        reward += 1
        reward -= 5 * sqrt(osh.d_chopcenterpos2obj)
    elseif osh.d_chopcenterpos2obj < 0.03
        # start clamping chopsticks
        reward += 0.03
        reward -= 0.1 * abs(osh.d_choptips - dclosed)
    else
        # open chopsticks
        reward -= 0.1 * abs(osh.d_choptips - dopen)
    end

    return reward
end

LyceumMuJoCo.geteval(state, action, obs, env::HEBIPickup) = 0

@inline LyceumMuJoCo.getsim(env::HEBIPickup) = env.sim


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

function iscentered(e::HEBIPickup)
    @unpack m, d, mn, dn = e.sim
    dx = dn.site_xpos[:, :chop] - dn.site_xpos[:, :stick]
    dx = dx / norm(dx)
    #pnt = dn.site_xpos[:, :stick] + dx * 0.003
    pnt = dn.site_xpos[:, :stick] + dx * 0.004
    geomid = Cint[-1]
    dist = mj_ray(m, d, pnt, dx, C_NULL, UInt8(0), -1, geomid)
    first(geomid) == 22
end

function chopstickcenter(e::HEBIPickup)
    @unpack m, d, mn, dn = e.sim
    dx = dn.site_xpos[:, :chop] - dn.site_xpos[:, :stick]
    dn.site_xpos[:, :stick] + dx / 2
end

function iscontacting(e::HEBIPickup)
    # stick=19, chop=20, block=22
    stick = false
    chop = false
    for c in e.sim.d.contact
        stick |= (c.geom1 == 19 && c.geom2 == 22 || c.geom1 == 22 && c.geom2 == 19)
        chop |= (c.geom1 == 20 && c.geom2 == 22 || c.geom1 == 22 && c.geom2 == 20)
        #(c.geom1 == 19 || c.geom1 == 20) && c.geom2 == 22 && return true
        #(c.geom2 == 19 || c.geom2 == 20) && c.geom1 == 22 && return true
    end
    return stick && chop
end

function ispickedup(e::HEBIPickup)
    for c in e.sim.d.contact
        c.geom1 == 22 && c.geom2 == 0 && return false
        c.geom1 == 0 && c.geom2 == 22 && return false
    end
    return iscontacting(e)
end
