using MultiSymplectic
struct Trial_Solution_Basis <: AbstractPDEBasis
    PNN
    trial_sol_type::Symbol  # :TFC or :Plain

    u
    v
    w
end

function Trial_Solution_Basis(PNN,problem,trial_sol_type=:TFC,previous_params=nothing)
    local h = problem.tstep
    local xspan = problem.xspan
    local a,b = problem.xspan[1],problem.xspan[2]
    local x_length = b - a
    local ic_fun = problem.ics_function
    local bc_fun = problem.bcs_function


    psi_L(x) = (b - x) / x_length       # left 
    psi_R(x) = (x - a) / x_length       # right
    phi_B(t,tn) = (tn + h - t) / h   # bottom


    function T1NN(t, x, tn, params)
        return psi_L(x) * PNN([t,a], params)[1] +
            psi_R(x) * PNN([t,b], params)[1] +
            phi_B(t,tn) * PNN([tn,x], params)[1]
    end

    function T2NN(t, x, tn, params)
        return psi_L(x) * phi_B(t,tn) * PNN([tn,a], params)[1] +
            psi_R(x) * phi_B(t,tn) * PNN([tn,b], params)[1]
    end

    function C1(t, x, tn,params)
        if tn == 0.0
            return psi_L(x) * ic_fun(xspan).u +
                psi_R(x) * ic_fun(xspan).u +
                phi_B(t,tn) * ic_fun(x).u #exact_u(tn,x) 
        else
            return psi_L(x) * bc_fun(t, xspan).bc₀.u +
                psi_R(x) * bc_fun(t, xspan).bc₁.u +
                phi_B(t,tn) * #PNN([tn+h,x], previous_params)[1] 
    end

    function C2(t, x, tn, params)
        return psi_L(x) * phi_B(t,tn) * bc_fun(tn, xspan).bc₀.u +
            psi_R(x) * phi_B(t,tn) * bc_fun(tn, xspan).bc₁.u
    end


    u_trial(t,x,tn,params) =PNN([t,x], params)[1] - T1NN(t,x,tn,params) + T2NN(t,x,tn,params) + C1(t,x,tn,params) - C2(t,x,tn,params)
    v_trial(t,x,tn,params) = Zygote.gradient(tt -> u_trial(tt,x,tn,params),t)[1]
    w_trial(t,x,tn,params) = Zygote.gradient(xx -> u_trial(t,xx,tn,params),x)[1]
    return Trial_Solution_Basis(PNN,trial_sol_type,u_trial,v_trial,w_trial)
end