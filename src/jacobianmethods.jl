function dls_jacctrl!(dtheta::AbsVec, jac::AbsMat, error::AbsVec; lambda::Real = 1.1)
    dtheta .= inv((jac' * jac + lambda^2 * I)) * jac' * error
end

pinv_jacctrl!(dtheta::AbsVec, jac::AbsMat, error::AbsVec) = dtheta .= pinv(jac) * error

function transpose_jacctrl!(dtheta::AbsVec, jac::AbsMat, error::AbsVec)
    dtheta .= transpose(jac) * error
end

function sdls_jacctrl!(dtheta::AbsVec, jac::AbsMat, error::AbsVec; lammax::Real = pi / 8)
    dofefc, nv = size(jac)
    u, s, v = svd(jac)

    N = zeros(dofefc)
    for i = 1:dofefc
        N[i] = norm(u[:, i])
    end

    M = zeros(dofefc)
    for i = 1:dofefc
        for j = 1:nv
            pj = norm(jac[:, j])
            M[i] += 1 / s[i] * abs(v[j, i]) * pj
        end
    end

    lam = zeros(dofefc)
    for i = 1:dofefc
        lam[i] = min(1, N[i] / M[i]) * lammax
    end

    phi = zeros(nv, dofefc)
    for i = 1:dofefc, j = 1:nv
        alpha = transpose(u[:, i]) * error
        phi[:, i] = clampnorm!(1 / s[i] * alpha * v[:, i], lam[i], 1)
    end

    for i = 1:dofefc
        dtheta .+= phi[:, i]
    end
    clampnorm!(dtheta, lammax, 1)

    return dtheta
end
