@inline function posinit!(meantraj)
    #@uviews meantraj @inbounds begin
    #    lastcontrols = view(meantraj, :, size(meantraj, 2))
    #    lastcontrols1 = view(meantraj, :, size(meantraj, 2) - 1)
    #    fill!(lastcontrols, last)
    #end
    @assert meantraj[:, size(meantraj, 2)] == meantraj[:, size(meantraj, 2) - 1]
end



sos(x) = sum(el -> el ^ 2, x)
function hebiMPPI(etype = HebiPickup)
    env = etype()

    a = getaction(env)
    #a[1:7]   .= 0.2
    #a[8:end] .= 0.0001

    posweight = 1 #0.7
    rotweight = 1 - posweight
    osp = obsspace(env)
    dmin = 0.025
    value = @closure (o) -> begin
        osh = osp(o)
        reward = 0.0
        #reward -= 10 * osh.d_chopcenterpos2obj
        #reward -= osh.d_obj2goal
        if osh.ispickedup == 1.0
            reward += 1
            #reward -= osh.d_obj2goal
        else
        #    reward -= 10
        end
        reward * 50
    end

    #a[1:7]   .= 0.2
    #a[8:end] .= 0.0001
    sigma = zeros(7)
    sigma[1:3] .= 0.15
    sigma[4:6] .= 0.15
    sigma[end] = 0.05
    mppi = MPPI(
        env_tconstructor = n -> tconstruct(etype, n),
        covar = Diagonal(sigma .^ 2),
        lambda = 0.05,

        H = 70,
        K = 16,
        gamma = 1.0,
        value = value,
    )

    a = allocate(actionspace(env))
    o = allocate(obsspace(env))
    s = allocate(statespace(env))

    i=0
    ctrlfn = @closure env -> begin

        #display(obsspace(env)(getobs(env)).d_chopcenterpos2obj < 0.005)
        #@info obsspace(env)(getobs(env)).d_obj

        #@unpack m, d = env.sim
        #dt = ifelse(d.ncon > 0, 0.001, 0.004)
        #@set! m.opt.timestep = dt
        #opt = m.opt
        #m.opt = LyceumBase.SetfieldImpl.Setfield.@set opt.timestep = dt

        #@info obsspace(env)(getobs(env)).d_choptips

        getstate!(s, env)
        getaction!(a, s, mppi)
        setaction!(env, a)
    end

    visualize(env, controller=ctrlfn)
end

