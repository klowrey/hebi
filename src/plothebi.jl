
function hebiroll(env, act)
    env.sim.d.qpos .= pos[:,1]
    env.sim.d.qvel .= vel[:,1]
    env.sim.d.ctrl .= act[:,1]
    forward!(env.sim)
    traj = _sysidrollout(env, act)
end

function plotdif_ref(env::HebiPickup, pos, vel, act::AbstractMatrix)
    reset!(env)

    env.sim.d.qpos .= pos[:,1]
    env.sim.d.qvel .= vel[:,1]
    env.sim.d.ctrl .= act[:,1]
    forward!(env.sim)
    traj = _sysidrollout(env, act)
    #ref = copy(traj.states)
    #ref[2:8, :]  .= pos
    #ref[9:15, :] .= vel
    #ref[16:end, :] .= 0.0

    #reset!(env)
    #visualize(env, trajectories=[ref, traj.states])
    #return traj.states[2:8,:], traj.states[9:15,:]
    return traj.obses[1:7,:], traj.obses[8:14,:]
end

function plotx8(env, pos, vel, act, r=1:size(pos,2), pr=1:3)
   npos, nvel = plotdif_ref(env, pos[:,r], vel[:,r], act[:,r])

   println("Poss")
   println(mse(pos[pr,r], npos[pr,:]))
   for i=pr
      display(lineplot!(
                        lineplot(npos[i,:], xlim=(0,length(r)), 
                                 width=80, height=7, name="est"),
                        pos[i,r], name="real"))
   end
   println("Vels")
   println(mse(vel[pr,r], nvel[pr,:]))
   for i=pr
      display(lineplot!(
                        lineplot(nvel[i,:], xlim=(0,length(r)), 
                                 width=80, height=7, name="est"),
                        vel[i,r], name="real"))
   end
end

function radps2rotpm(r)
   r * 60 / 2pi
end
function rotpm2radps(r)
   r * 2pi / 60
end

max_torque = SVector{7, Float64}(20.0, 38.0, 20.0, 2.5, 2.5, 2.5, 2.5)
speed_24v = SVector{7, Float64}(3.267, 1.759, 3.267, 14.074, 14.074, 14.074, 14.074) # rad/sec / volt * 24v
vellimits = SVector{7, Float64}(3.1416, 1.5708, 3.1416, 9.4248, 9.4248, 9.4248, 9.4248) # rad / sec
slope     = SVector{7, Float64}(-6.369, -24.204, -6.369, -0.2132, -0.2132, -0.2132, -0.2132) # max_torque / (speed_24v)

term_resist = [3.19, 3.19, 3.19, 9.99, 9.99, 9.99, 9.99] 

function testctrl(h, act, vel, eff, i)

   a = act[i,:]
   s = max_torque[i] / speed_24v[i]

   # TODO recalculate the speed_24v with new 60% velocity limits; slope will change.

   #println(s, " ", slope[i])
   #t = @. slope[i] * vel[i,:] + act[i,:] * max_torque[i]
   #maxT = @. -s * abs(vel[i,:]) + max_torque[i] # where we are on the speed-torque curve
   maxT = @. s * vel[i,:] + max_torque[i] # where we are on the speed-torque curve
   #maxT = @. -s * vel[i,:] + max_torque[i] # where we are on the speed-torque curve
   t = @. act[i,:] * maxT #- (s * abs(vel[i,:]))

   return t
end


#dt, pos, vel, act, eff = datafile("data/wrist1-v.csv", 7);
#i = 4; p!(p(eff[i,:], width=140, xlim=(80,40000)), testctrl(h, act, vel, eff, i))
