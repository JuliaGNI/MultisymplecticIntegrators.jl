struct Trial_Solution_Basis <: AbstractPDEBasis
    PNN
    trial_sol_type::Symbol  # :TFC or :Plain

    u
    v
    w
end

function Trial_Solution_Basis(PNN,problem,trial_sol_type=:TFC)
    local h = problem.tstep
    local xspan = problem.xspan
    local a,b = problem.xspan[1],problem.xspan[2]
    local x_length = b - a
    local icsf = problem.ics_function
    local bcsf = problem.bcs_function

    # t is the quadrature point without scaling
    T₁NN(t,x,tn,params) = ((tn + h -t)/h)*PNN([tn,x],params)[1] + ((b-x)/(b-a))*PNN([t,a],params)[1] + ((x-a)/(b-a))*PNN([t,b],params)[1]
    T₂NN(t,x,tn,params) = ((tn + h -t)*(b-x)/(x_length * h))*PNN([tn,a],params)[1] + ((tn + h -t)*(x-a)/(x_length * h))*PNN([tn,b],params)[1]
    C₁(t,x,tn) = ((tn + h -t)/h)*icsf(x).u + ((b-x)/(b-a))*bcsf(t,xspan).bc₀.u + ((x-a)/(b-a))*bcsf(t,xspan).bc₁.u
    C₂(t,x,tn) = ((tn + h -t)*(b-x)/(x_length * h))*bcsf(tn,xspan).bc₀.u + ((tn + h -t)*(x-a)/(x_length * h))*bcsf(tn,xspan).bc₁.u

    u_trial(t,x,tn,params) =PNN([t,x],params)[1]+ T₁NN(t,x,tn,params) - T₂NN(t,x,tn,params) + C₁(t,x,tn) - C₂(t,x,tn)
    v_trial(t,x,tn,params) = Zygote.gradient(tt -> u_trial(tt,x,tn,params),t)[1]
    w_trial(t,x,tn,params) = Zygote.gradient(xx -> u_trial(t,xx,tn,params),x)[1]
    return Trial_Solution_Basis(PNN,trial_sol_type,u_trial,v_trial,w_trial)
end