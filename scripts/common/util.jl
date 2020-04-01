function rollout(e::AbstractEnvironment, actions::AbstractMatrix)
    T = size(actions, 2)

    traj = (
        states = Array(undef, statespace(e), T),
        obses = Array(undef, obsspace(e), T),
        acts = actions,
        rews = Array(undef, rewardspace(e), T),
    )

    for t = 1:T
        st = view(traj.states, :, t)
        at = view(traj.acts, :, t)
        ot = view(traj.obses, :, t)

        getstate!(st, e)
        getobs!(ot, e)
        setaction!(e, at)
        step!(e)
        traj.rews[t] = getreward(st, at, ot, e)
    end

    return traj
end
