struct PDController{L,T}
    kp::SVector{L,T}
    kd::SVector{L,T}
    function PIDController{L,T}(kp, kd) where {L,T}
        new{L,T}(kp, kd)
    end
end

function Lyceum.getaction!(a, veltarget, vel, acc, pd::PDController)
    a .= a.kp * poserror + a.kd * vel
    a
end