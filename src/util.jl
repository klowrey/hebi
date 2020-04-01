function randset!(h2, h1)
    h2.sim.m.dof_damping[1:3] .= h1.sim.m.dof_damping[1] .+ rand(Uniform(-0.05, 4.5), 3)
    h2.sim.m.dof_damping[4:7] .= h1.sim.m.dof_damping[4] .+ rand(Uniform(-0.05, 4.5), 4)
    h2.sim.m.dof_armature[1:3] .= h1.sim.m.dof_armature[1] .+ rand(Uniform(0, 10.2), 3)
    h2.sim.m.dof_armature[4:7] .= h1.sim.m.dof_armature[4] .+ rand(Uniform(0, 10.2), 4)
    h2.sim.m.dof_frictionloss[1:3] .= h1.sim.m.dof_frictionloss[1] .+ rand(Uniform(-0.05, 3.5), 3)
    h2.sim.m.dof_frictionloss[4:7] .= h1.sim.m.dof_frictionloss[4] .+ rand(Uniform(-0.05, 3.5), 4)
    h2
end
function resetset!(h2, h1)
    h2.sim.m.dof_damping .= h1.sim.m.dof_damping
    h2.sim.m.dof_armature .= h1.sim.m.dof_armature
    h2.sim.m.dof_frictionloss .= h1.sim.m.dof_frictionloss
end

function _sysidtraj(e::AbstractEnvironment, T::Integer)
    (
        states = Array(undef, statespace(e), T),
        obses = Array(undef, obsspace(e), T),
        #acts = Array(undef, actionspace(e), T)
    )
end

function _sysidrollout(
    e::AbstractEnvironment,
    actions::AbstractMatrix,
    traj = _sysidtraj(e, size(actions, 2)),
)
    T = size(actions, 2)
    #traj.acts .= actions
    for t = 1:T
        st = view(traj.states, :, t)
        #at = view(traj.acts, :, t)
        at = view(actions, :, t)
        ot = view(traj.obses, :, t)

        getstate!(st, e)
        getobs!(ot, e)
        setaction!(e, at)
        step!(e)
    end
    traj
end


function vizdif_ref(env::HEBIPickup, pos, vel, ctrls::AbstractMatrix)
    reset!(env)

    env.sim.d.qpos .= pos[:, 1]
    env.sim.d.qvel .= vel[:, 1]

    traj = _rollout(env, ctrls)
    ref = copy(traj.states)
    ref[2:8, :] .= pos
    ref[9:15, :] .= vel
    ref[16:end, :] .= 0.0

    #reset!(env)
    visualize(env, trajectories = [ref, traj.states])
    return traj.states[2:8, :], traj.states[9:15, :]
end

function vizdif(env::HEBIPickup, env2::HEBIPickup, ctrls::AbstractMatrix)
    reset!(env)
    traj = _rollout(env, ctrls)
    reset!(env2)
    data = _rollout(env2, ctrls)

    println(mse(data.obses, traj.obses))
    visualize(env, trajectories = [traj.states, data.states])
end


function clampnorm!(A, maxnorm::Real, p::Real = 2)
    Anorm = norm(A, p)
    if Anorm > maxnorm
        A .= maxnorm .* A ./ Anorm
    end
    A
end

function clampmaxabs!(A, maxnorm::Real)
    Anorm = maximum(abs, A)
    if Anorm > maxnorm
        A .= maxnorm .* A ./ Anorm
    end
    A
end


function mj_ray(
    m::Model,
    d::Data,
    pnt::MJVec{mjtNum},
    vec::MJVec{mjtNum},
    geomgroup::Union{MJVec{mjtByte},Ptr{Nothing}},
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
