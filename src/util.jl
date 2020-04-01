function clampnorm!(A, maxnorm::Real, p::Real = 2)
    Anorm = norm(A, p)
    if Anorm > maxnorm
        A .= maxnorm .* A ./ Anorm
    end
    return A
end

function clampmaxabs!(A, maxnorm::Real)
    Anorm = maximum(abs, A)
    if Anorm > maxnorm
        A .= maxnorm .* A ./ Anorm
    end
    return A
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
