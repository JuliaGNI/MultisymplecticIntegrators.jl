struct TrialNN_PDE_int{BT<:AbstractPDEBasis,IPMT<:InitialParametersMethod} <: PDEMethod
    basis::BT

    time_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RT::Int # Number of quadrature points in time

    spatial_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions

    grid_matrix::Matrix{Vector{Float64}}  # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights::Matrix{Float64} # Quadrature weights

    initial_guess_method::IPMT # :LSGD or :GroundTruth

    Nw::Int                      # angular directions
    Nb::Int                      # bias samples

    show_status::Bool
    xspan::Tuple{Float64, Float64}
    function TrialNN_PDE_int(basis,;RT_per_interval::Int = 4,RX_per_interval::Int = 4,
        nx::Int = 40,nt::Int= 20,Nw::Int=500, Nb::Int=500,show_status::Bool = false,
        xspan::Tuple=(0., 1.0),t_num_interval::Int=2,x_num_interval::Int=5,
        initial_guess_method::IPMT=OGA2D(xspan[1], xspan[2],basis.activation_function,nx = nx,nt = nt,Nw = Nw,Nb = Nb),) where {IPMT,} # 300,300

        t_quadrature = composite_quadrature(t_num_interval ,RT_per_interval)
        x_quadrature = composite_quadrature(x_num_interval ,RX_per_interval)

        RT = length(t_quadrature.nodes)
        RX = length(x_quadrature.nodes)

        R_list = [RT_per_interval,RX_per_interval]
        grid_matrix, grid_weights = construct_quadrature_grid(R_list,[t_num_interval, x_num_interval])
        # scale grid_matrix
        x0 = xspan[1]
        x_domain = xspan[2] - xspan[1]

        grid_matrix = [collect(grid_matrix[i, j]) for i in axes(grid_matrix, 1), j in axes(grid_matrix, 2)]

        @inbounds for k in eachindex(grid_matrix)
            t, xhat = grid_matrix[k]
            grid_matrix[k][2] = x0 + x_domain * xhat
        end

        new{typeof(basis),typeof(initial_guess_method)}(basis,
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            initial_guess_method,
            Nw,Nb,
            show_status,xspan)
    end
end

function Base.show(io::IO, method::TrialNN_PDE_int)
    print(io, "\n Trial Neural Network PDE Integrator with:\n")
    print(io, "   Basis: $(nameof(typeof(method.basis))) \n")
    print(io, "   Basis parameters NP: $(method.basis.NP), hidden units S: $(method.basis.S) \n")
    print(io, "   Time quadrature points: $(method.RT), space quadrature points: $(method.RX) \n")
    print(io, "   Space span: $(method.xspan) \n")
    print(io, "   Initial guess method: $(nameof(typeof(method.initial_guess_method))) \n")
    print(io, "   OGA dictionary sizes: Nw=$(method.Nw), Nb=$(method.Nb) \n")
    print(io, "   Show status: $(method.show_status) \n")
end

default_solver(::TrialNN_PDE_int) = NewtonMethod()
default_iguess(::TrialNN_PDE_int) = nothing
struct TrialNN_PDE_intCache{ST,RT,RX,D,S,NP,OGAN,a,b,xstep} <: PDEIntegratorCache{ST,D}
    """
    RT = number of quadrature points in time
    RX = number of quadrature points in space
    D = dimension of output of u,v,w, i.e. scaler value function (D = 1) or vector function
    NP = number of parameters in the expression
    """
    x::Vector{ST}
    W2::Vector{ST}
    W1::Matrix{ST}
    bias1::Vector{ST}

    previous_W2::Vector{ST}
    previous_W1::Matrix{ST}
    previous_bias1::Vector{ST}

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

    ∂u∂θ_quad_values::Array{ST}
    ∂v∂θ_quad_values::Array{ST}
    ∂w∂θ_quad_values::Array{ST}
    flag_done_initial_guess::Vector{ST}

    C1C2_equispaced_quad_nodes::Vector{ST}
    C1C2_quad::Array{ST}
    ∂C1C2∂t_quad::Array{ST}
    ∂C1C2∂x_quad::Array{ST}

    x_nodes::Vector{ST}
    C1C2_result::Matrix{ST}
    ∂C1C2∂t_result::Matrix{ST}
    ∂C1C2∂x_result::Matrix{ST}

    previous_C1C2_equispaced_quad_nodes::Vector{ST}
    previous_C1C2_quad::Array{ST}
    previous_∂C1C2∂t_quad::Array{ST}
    previous_∂C1C2∂x_quad::Array{ST}

    previous_C1C2_result::Matrix{ST}
    previous_∂C1C2∂t_result::Matrix{ST}
    previous_∂C1C2∂x_result::Matrix{ST}

    function TrialNN_PDE_intCache{ST,RT,RX,D,S,NP,OGAN,a,b,xstep}() where {ST,RT,RX,D,S,NP,OGAN,a,b,xstep}
        x = zeros(ST, NP) # in ELM, x is just the output layer parameters

        W2 = zeros(ST, S)
        W1 = zeros(ST, S, 2)
        bias1 = zeros(ST, S)

        previous_W2 = zeros(ST, S)
        previous_W1 = zeros(ST, S, 2)
        previous_bias1 = zeros(ST, S)

        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        ∂u∂θ_quad_values = zeros(ST, D, RT, RX, NP)
        ∂v∂θ_quad_values = zeros(ST, D, RT, RX, NP)
        ∂w∂θ_quad_values = zeros(ST, D, RT, RX, NP)
        flag_done_initial_guess = zeros(ST, 1)

        C1C2_equispaced_quad_nodes = zeros(ST,OGAN)

        C1C2_quad=zeros(ST, D, RT, RX)
        ∂C1C2∂t_quad=zeros(ST, D, RT, RX)
        ∂C1C2∂x_quad=zeros(ST, D, RT, RX)

        x_nodes = collect(a:xstep:b)
        C1C2_result=zeros(ST, D, length(x_nodes))
        ∂C1C2∂t_result=zeros(ST, D, length(x_nodes))
        ∂C1C2∂x_result=zeros(ST, D, length(x_nodes))

        previous_C1C2_equispaced_quad_nodes = zeros(ST, OGAN)

        previous_C1C2_quad=zeros(ST, D, RT, RX)
        previous_∂C1C2∂t_quad=zeros(ST, D, RT, RX)
        previous_∂C1C2∂x_quad=zeros(ST, D, RT, RX)

        previous_C1C2_result=zeros(ST, D, length(x_nodes))
        previous_∂C1C2∂t_result=zeros(ST, D, length(x_nodes))
        previous_∂C1C2∂x_result=zeros(ST, D, length(x_nodes))



        new(x,
            W2,W1,bias1,
            previous_W2,previous_W1,previous_bias1,
            u_quad_values,v_quad_values,w_quad_values,
            ∂L∂U_quad_values,∂L∂V_quad_values,∂L∂W_quad_values,
            ∂u∂θ_quad_values,∂v∂θ_quad_values,∂w∂θ_quad_values,
            flag_done_initial_guess,
            C1C2_equispaced_quad_nodes,C1C2_quad,
            ∂C1C2∂t_quad,∂C1C2∂x_quad,
            x_nodes,
            C1C2_result,∂C1C2∂t_result,∂C1C2∂x_result,
            previous_C1C2_equispaced_quad_nodes,previous_C1C2_quad,
            previous_∂C1C2∂t_quad,previous_∂C1C2∂x_quad,
            previous_C1C2_result,previous_∂C1C2∂t_result,previous_∂C1C2∂x_result
            )
    end
end

nlsolution(cache::TrialNN_PDE_intCache) = cache.x

function Cache{ST}(problem::LPDEProblem, int::TrialNN_PDE_int; kwargs...) where {ST}
    TrialNN_PDE_intCache{ST,int.RT,int.RX,problem.D,int.basis.S,int.basis.NP,int.initial_guess_method.N,int.xspan[1],int.xspan[2],problem.xstep}(; kwargs...)
end

@inline CacheType(ST, problem::LPDEProblem, int::TrialNN_PDE_int) = TrialNN_PDE_intCache{ST,int.RT,int.RX,problem.D,int.basis.S,int.basis.NP,int.initial_guess_method.N,int.xspan[1],int.xspan[2],problem.xstep}

function NN(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}) where {TT,XT, DT}
    local activation = int.method.basis.activation_function
    return sum(W2[i] * activation(W1[i,1]*t + W1[i,2]*x + bias1[i]) for i in eachindex(W2))
end

function T1NN_manual(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}) where {TT,XT, DT}
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local h = timestep(int)
    local x_domain = b-a

    return (b - x) / x_domain * NN(t, a, W2,W1,bias1,int) +
           (x - a) / x_domain * NN(t, b, W2,W1,bias1,int) +
           (h - h * t) / h * NN(0.0, x, W2,W1,bias1,int)
end

function T2NN_manual(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}) where {TT,XT, DT}
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local h = timestep(int)
    local x_domain = b-a

    return (b - x) / x_domain * (h - h * t) / h * NN(0.0, a, W2,W1,bias1,int) +
           (x - a) / x_domain * (h - h * t) / h * NN(0.0, b, W2,W1,bias1,int)
end

function C1(t::TT, x::XT, tn::ST, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {TT,XT,ST,SLT}
    local xspan = int.problem.xspan
    local a,b = xspan[1],xspan[2]
    local x_domain = b-a
    local exact_u = int.problem.exact_u
    local h = timestep(int)
    local W1 = cache(int).previous_W1
    local bias1 = cache(int).previous_bias1
    local W2 = cache(int).previous_W2
    local u_func = int.method.basis.u_func
    local S = int.method.basis.S

    all_previous_params = vcat(
        W2,
        vec(W1),
        bias1
    )

    if tn == 0.0
        return (b - x) * exact_u(h*t, a) / x_domain +
           (x - a) * exact_u(h*t, b) / x_domain +
           (h - h * t) * exact_u(tn, x) / h
    else
        # return (b - x) * exact_u(tn+h*t, a) / x_domain +(x - a) * exact_u(tn+h*t, b) / x_domain + (h - h * t) * u_trial(1.0, x, W2,W1,bias1,int,sol) / h # should be u_trial
        return (b - x) * exact_u(tn+h*t, a) / x_domain +(x - a) * exact_u(tn+h*t, b) / x_domain + (h - h * t) * u_func(1.0, x, h, all_previous_params,) / h # should be u_trial

    end
end

function C2(t::TT, x::XT, tn::ST, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {TT,XT,ST,SLT}
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local h = timestep(int)
    local exact_u = int.problem.exact_u

    return (b - x) * (h - h * t) * exact_u(tn, a) / x_domain / h +
           (x - a) * (h - h * t) * exact_u(tn, b) / x_domain / h
end

function u_trial(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {TT,XT, DT,SLT}
    local tn = sol.t - timestep(int)
    NN(t, x, W2, W1, bias1, int) - T1NN_manual(t, x, W2, W1, bias1, int) + T2NN_manual(t, x, W2, W1, bias1, int)  #+ C1(t, x, tn, int) - C2(t, x, tn, int)
end

function u_trial(t::TT,x::XT,all_params::AbstractVector{DT},int::PDEIntegrator{<:TrialNN_PDE_int},sol::SLT) where {TT,XT, DT,SLT}
    local S = int.method.basis.S
    W2 = all_params[1:S]
    W1 = reshape(all_params[S+1:3*S], S, 2)
    bias1 = all_params[3*S+1:4*S]
    return u_trial(t,x,W2,W1,bias1,int,sol)
end

function C1C2(t::TT, x::XT, tn::Float64, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {TT,XT,SLT}
    C1(t, x, tn, int,sol) - C2(t, x, tn, int,sol)
end

C1C2(tx::Vector{ST},tn::Float64,int::PDEIntegrator{<:TrialNN_PDE_int},sol::SLT) where {ST,SLT} = C1C2(tx[1],tx[2],tn,int,sol)

∂C1C2∂t(t::TT, x::XT, tn::Float64, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {TT,XT,SLT} = Zygote.gradient(tt -> C1C2(tt,x,tn,int,sol),t)[1]
∂C1C2∂x(t::TT, x::XT, tn::Float64, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {TT,XT,SLT} = Zygote.gradient(xx -> C1C2(t,xx,tn,int,sol),x)[1]

∂C1C2∂t(tx::Vector{ST},tn::Float64,int::PDEIntegrator{<:TrialNN_PDE_int},sol::SLT) where {ST,SLT} = ∂C1C2∂t(tx[1],tx[2],tn,int,sol)
∂C1C2∂x(tx::Vector{ST},tn::Float64,int::PDEIntegrator{<:TrialNN_PDE_int},sol::SLT) where {ST,SLT} = ∂C1C2∂x(tx[1],tx[2],tn,int,sol)

u_trial(tx::Vector{ST},all_params::Vector{DT},int::PDEIntegrator{<:TrialNN_PDE_int},sol::SLT) where {ST, DT,SLT} = u_trial(tx[1],tx[2],all_params,int,sol)
u_trial(tx::NTuple{2,ST},all_params::Vector{DT},int::PDEIntegrator{<:TrialNN_PDE_int},sol::SLT) where {ST<:Real, DT,SLT} = u_trial(tx[1],tx[2],all_params,int,sol)
v_trial_zygote(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {TT,XT, DT,SLT} = Zygote.gradient(tt -> u_trial(tt,x,W2,W1,bias1,int,sol),t)[1]
w_trial_zygote(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {TT,XT, DT,SLT} = Zygote.gradient(xx -> u_trial(t,xx,W2,W1,bias1,int,sol),x)[1]


u_trial(tx::Vector{ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST, DT,SLT} = u_trial(tx[1],tx[2],W2,W1,bias1,int,sol)
u_trial(tx::NTuple{2,ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST<:Real, DT,SLT} = u_trial(tx[1],tx[2],W2,W1,bias1,int,sol)
v_trial_zygote(tx::Vector{ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST, DT,SLT} = v_trial_zygote(tx[1],tx[2],W2,W1,bias1,int,sol)
v_trial_zygote(tx::NTuple{2,ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST<:Real, DT,SLT} = v_trial_zygote(tx[1],tx[2],W2,W1,bias1,int,sol)
w_trial_zygote(tx::Vector{ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST, DT,SLT} = w_trial_zygote(tx[1],tx[2],W2,W1,bias1,int,sol)
w_trial_zygote(tx::NTuple{2,ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST<:Real, DT,SLT} = w_trial_zygote(tx[1],tx[2],W2,W1,bias1,int,sol)

# v_trial(t, x, W2,W1,bias1,int,sol) = (1 / timestep(int)) * ForwardDiff.derivative(tt -> u_trial(tt,x,W2,W1,bias1,int,sol),t)[1]
# w_trial(t, x, W2,W1,bias1,int,sol) = ForwardDiff.derivative(xx -> u_trial(t,xx,W2,W1,bias1,int,sol),x)[1]

# v_trial_zygote(t, x, all_params,int,sol) = (1 / timestep(int)) * Zygote.gradient(tt -> u_trial(tt,x,all_params,int,sol),t)[1]
# w_trial_zygote(t, x, all_params,int,sol) = Zygote.gradient(xx -> u_trial(t,xx,all_params,int,sol),x)[1]

v_trial(t::TT,x::XT,params::AbstractVector{DT},int::PDEIntegrator{<:TrialNN_PDE_int},sol::SLT) where {TT,XT, DT,SLT} = ForwardDiff.derivative(tt -> u_trial(tt,x,params,int,sol),t)[1]
w_trial(t::TT,x::XT,params::AbstractVector{DT},int::PDEIntegrator{<:TrialNN_PDE_int},sol::SLT) where {TT,XT, DT,SLT} = ForwardDiff.derivative(xx -> u_trial(t,xx,params,int,sol),x)[1]


∂u∂p(t::TT,x::XT,params::AbstractVector{DT},int::PDEIntegrator{<:TrialNN_PDE_int},sol::SLT) where {TT,XT, DT,SLT} = ForwardDiff.gradient(p -> u_trial(t,x,p,int,sol),params)
∂v∂p(t::TT,x::XT,params::AbstractVector{DT},int::PDEIntegrator{<:TrialNN_PDE_int},sol::SLT) where {TT,XT, DT,SLT} = ForwardDiff.gradient(p -> v_trial(t,x,p,int,sol),params)
∂w∂p(t::TT,x::XT,params::AbstractVector{DT},int::PDEIntegrator{<:TrialNN_PDE_int},sol::SLT) where {TT,XT, DT,SLT} = ForwardDiff.gradient(p -> w_trial(t,x,p,int,sol),params)

∂u∂p(input::Vector{ST}, params::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST, DT,SLT} = ∂u∂p(input[1],input[2],params,int,sol)
∂v∂p(input::Vector{ST}, params::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST, DT,SLT} = ∂v∂p(input[1],input[2],params,int,sol)
∂w∂p(input::Vector{ST}, params::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST, DT,SLT} = ∂w∂p(input[1],input[2],params,int,sol)
∂u∂p(input::NTuple{2,ST}, params::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST<:Real, DT,SLT} = ∂u∂p(input[1],input[2],params,int,sol)
∂v∂p(input::NTuple{2,ST}, params::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST<:Real, DT,SLT} = ∂v∂p(input[1],input[2],params,int,sol)
∂w∂p(input::NTuple{2,ST}, params::Vector{DT}, int::PDEIntegrator{<:TrialNN_PDE_int}, sol::SLT) where {ST<:Real, DT,SLT} = ∂w∂p(input[1],input[2],params,int,sol)

# ∂u∂W2(t, x, W2,W1,bias1,int,sol) = ForwardDiff.gradient(p -> u_trial(t,x,p,W1,bias1,int,sol),W2)
# ∂v∂W2(t, x, W2,W1,bias1,int,sol) = (1 / timestep(int)) * ForwardDiff.gradient(p -> v_trial(t,x,p,W1,bias1,int,sol),W2)
# ∂w∂W2(t, x, W2,W1,bias1,int,sol) = ForwardDiff.gradient(p -> w_trial(t,x,p,W1,bias1,int,sol),W2)

# ∂u∂W1(t, x, W2,W1,bias1,int,sol) = ForwardDiff.gradient(p -> u_trial(t,x,W2,p,bias1,int,sol),W1)
# ∂v∂W1(t, x, W2,W1,bias1,int,sol) = (1 / timestep(int)) * ForwardDiff.gradient(p -> v_trial(t,x,W2,p,bias1,int,sol),W1)
# ∂w∂W1(t, x, W2,W1,bias1,int,sol) = ForwardDiff.gradient(p -> w_trial(t,x,W2,p,bias1,int,sol),W1)

# ∂u∂bias1(t, x, W2,W1,bias1,int,sol) = ForwardDiff.gradient(p -> u_trial(t,x,W2,W1,p,int,sol),bias1)
# ∂v∂bias1(t, x, W2,W1,bias1,int,sol) = (1 / timestep(int)) * ForwardDiff.gradient(p -> v_trial(t,x,W2,W1,p,int,sol),bias1)
# ∂w∂bias1(t, x, W2,W1,bias1,int,sol) = ForwardDiff.gradient(p -> w_trial(t,x,W2,W1,p,int,sol),bias1)




function mse_loss(params, tx_in, u_trial,int,sol)
    local current_step = sol.current_step
    local h = timestep(int)
    local tn = (current_step-1)*h
    local exact_u = int.problem.exact_u

    loss = 0.0
    for i in 1:size(tx_in, 2)
        t_samples = tx_in[1, i]
        x_samples = tx_in[2, i]
        pred = u_trial(t_samples, x_samples, params,int,sol)
        label = exact_u(tn + h * t_samples, x_samples)
        loss += (pred - label)^2
    end
    return loss / size(tx_in, 2)
end

function prior_initial_guess!(C::TrialNN_PDE_intCache, sol, int::PDEIntegrator{<:TrialNN_PDE_int{BT,IPMT}}) where {BT,IPMT<:PINN}
    local xspan = int.problem.xspan

    local PNN = int.method.basis.sol_network
    local BNN = int.method.basis.basis_network

    #use exact_sol for initialization temporarily for proof of concept
    # 5000 random points inside the domain
    tx_in = rand(Random.seed!(1),2,5000)
    tx_in[2,:] .= xspan[1] .+ (xspan[2] - xspan[1]) * tx_in[2,:]

    epochs = 100
    opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.AdamOptimizer(0.01), PNN.params)
    λ = GeometricMachineLearning.GlobalSection(PNN.params)
    loss_history = []

    batch_size = 100
    num_samples = size(tx_in, 2)
    num_batches = cld(num_samples, batch_size)
    print("Start initial guess function! \n")

    for epoch in 1:epochs
        epoch_loss = 0.0
        for batch_idx in 1:num_batches
            batch_start = (batch_idx - 1) * batch_size + 1
            batch_end = min(batch_idx * batch_size, num_samples)
            batch_tx = tx_in[:, batch_start:batch_end]

            grads = Zygote.gradient(d -> mse_loss(d, batch_tx,u_trial,int,sol), PNN.params)[1]
            GeometricMachineLearning.optimization_step!(opt, λ, PNN.params, grads)
            batch_loss = mse_loss(PNN.params, batch_tx,u_trial,int,sol)
            epoch_loss += batch_loss * size(batch_tx, 2)
        end
        epoch_loss /= num_samples
        push!(loss_history, epoch_loss)
        println("Epoch $epoch, MSE Loss: $epoch_loss")
        if epoch_loss < 1e-6
            println("Early stopping at epoch $epoch with loss $epoch_loss")
            break
        end
    end

    for (name, layer) in zip(keys(PNN.params), values(PNN.params))
        if hasfield(typeof(layer), :b)
            C.basis_nn_ps[name].W[:] = layer.W[:]
            C.basis_nn_ps[name].b[:] = layer.b[:]
            BNN.params[name].W[:] = layer.W[:]
            BNN.params[name].b[:] = layer.b[:]
        else
            # For layers without bias (e.g., output), just regenerate W
            C.x[:] = layer.W[:]
        end
    end
    println("Initial guess training completed.")
    println("Initial guess \n", C.x)


end

function prior_initial_guess!(C::TrialNN_PDE_intCache, sol, int::PDEIntegrator{<:TrialNN_PDE_int{BT,IPMT}}) where {BT,IPMT<:OGA2D}
    local h = timestep(int)

    local S = int.method.basis.S
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local exact_u = int.problem.exact_u
    local x_domain = b - a
    local tn = sol.t - timestep(int)
    local quad_nodes = int.method.initial_guess_method.equispaced_quad_nodes
    local quad_weights = int.method.initial_guess_method.quad_weights
    local A_mat = int.method.initial_guess_method.A_mat
    local Φ_raw = int.method.initial_guess_method.Φ_raw
    local N = int.method.initial_guess_method.N
    local M = int.method.initial_guess_method.M
    local C1C2_equispaced_quad_nodes = cache(int).C1C2_equispaced_quad_nodes

    # This performs up to `max_iter` outer iterations to account for boundary terms depending on PNN
    B = zeros(N, S)   # orthonormal basis columns
    coeffs_full = zeros(S)     # coefficients to write into PNN L2
    Wsel = zeros(S, 2)
    Bsel = zeros(S)
    desired = zeros(N)
    corrs = zeros(M)
    selected = zeros(Int, S) # indices of selected atoms in the dictionary

    # Build the desired internal PNN output on all quadrature nodes:
    # desired = target + T1NN - T2NN - C1 + C2  (evaluated with current PNN.params)
    for i in 1:N
        t = quad_nodes[1,i]; x = quad_nodes[2,i]
        desired[i] = exact_u(tn+h * t, x) - u_trial(t, x, coeffs_full,Wsel,Bsel,int,sol) - C1C2_equispaced_quad_nodes[i]
    end
    # @show desired

    # Run OGA (orthogonal matching) on Φ_raw to approximate `desired`
    residual = copy(desired)
    # @infiltrate
    for s = 1:S
        # compute correlations with residual (weighted)
        for i in 1:M
            corrs[i] = abs(sum(Φ_raw[i, :] .* (residual .* quad_weights)))
        end

        idx = argmax(corrs)
        selected[s] = idx
        length(Set(selected)) -1 == s ? nothing : @warn "atom repeated at s=$s, idx=$idx"

        # extract raw atom (already normalized) and orthogonalize (Gram-Schmidt)
        φ = copy(Φ_raw[idx, :])

        # append to B
        @views B[:, s] .= φ

        # solve least-squares for coefficients in orthonormal basis
        coeffs = view(B,:,1:s) \ desired         # small system k×1 solved implicitly
        # update residual
        residual = desired - view(B,:,1:s) * coeffs

        # store selection params (note A_mat rows correspond to atoms prior to normalization,
        # yet we normalized Φ_raw; we must store original (w,b) for a neuron consistent with A_mat)
        @views Wsel[s, :] .= A_mat[idx, 1:2]
        @views Bsel[s] = A_mat[idx, 3]

        coeffs_full[1:s] .= coeffs
        println("s=$s idx=$idx ‖residual‖=$(norm(residual))")
    end

    for j = 1:S
        @views C.W1[j, :] .= Wsel[j, :]
        C.bias1[j] = Bsel[j]
        C.W2[j] = coeffs_full[j]
        # C.x[j] = coeffs_full[j]
    end
    C.x[1:S] .= coeffs_full
    C.x[S+1:2*S] = Wsel[:,1]
    C.x[2*S+1:3*S] = Wsel[:,2]
    C.x[3*S+1:4*S] = Bsel[:]

    @show length(Set(selected)) == S  # number of unique selected atoms

    target_vec = [exact_u(tn+h*quad_nodes[1,i], quad_nodes[2,i]) for i in 1:N ]
    approx_vec = [u_trial(quad_nodes[1,i], quad_nodes[2,i], C.W2,C.W1,C.bias1,int,sol)+ C1C2_equispaced_quad_nodes[i] for i in 1:N ]
    err_vec = abs.(target_vec .- approx_vec)
    println("Max abs error after OGA initial guess: ", maximum(err_vec))

    # println("OGA initial guess completed.")
    # println("Initial guess \n", C.x)
end

function copy_internal_variables!(C::TrialNN_PDE_intCache,solstep::SolutionStep)
    haskey(internal(solstep), :previous_W2) && copyto!(C.previous_W2,internal(solstep).previous_W2)
    haskey(internal(solstep), :previous_W1) && copyto!(C.previous_W1,internal(solstep).previous_W1)
    haskey(internal(solstep), :previous_bias1) && copyto!(C.previous_bias1,internal(solstep).previous_bias1)

    haskey(internal(solstep), :previous_C1C2_equispaced_quad_nodes) && copyto!(C.previous_C1C2_equispaced_quad_nodes,internal(solstep).previous_C1C2_equispaced_quad_nodes)
    haskey(internal(solstep), :previous_C1C2_quad) && copyto!(C.previous_C1C2_quad,internal(solstep).previous_C1C2_quad)
    haskey(internal(solstep), :previous_∂C1C2∂t_quad) && copyto!(C.previous_∂C1C2∂t_quad,internal(solstep).previous_∂C1C2∂t_quad)
    haskey(internal(solstep), :previous_∂C1C2∂x_quad) && copyto!(C.previous_∂C1C2∂x_quad,internal(solstep).previous_∂C1C2∂x_quad)
    haskey(internal(solstep), :previous_C1C2_result) && copyto!(C.previous_C1C2_result,internal(solstep).previous_C1C2_result)
    haskey(internal(solstep), :previous_∂C1C2∂t_result) && copyto!(C.previous_∂C1C2∂t_result,internal(solstep).previous_∂C1C2∂t_result)
    haskey(internal(solstep), :previous_∂C1C2∂x_result) && copyto!(C.previous_∂C1C2∂x_result,internal(solstep).previous_∂C1C2∂x_result)

end

function copy_internal_variables!(solstep::SolutionStep,C::TrialNN_PDE_intCache)
    haskey(internal(solstep), :previous_W2) && copyto!(internal(solstep).previous_W2,C.W2)
    haskey(internal(solstep), :previous_W1) && copyto!(internal(solstep).previous_W1,C.W1)
    haskey(internal(solstep), :previous_bias1) && copyto!(internal(solstep).previous_bias1,C.bias1)

    haskey(internal(solstep), :previous_C1C2_equispaced_quad_nodes) && copyto!(internal(solstep).previous_C1C2_equispaced_quad_nodes,C.C1C2_equispaced_quad_nodes)
    haskey(internal(solstep), :previous_C1C2_quad) && copyto!(internal(solstep).previous_C1C2_quad,C.C1C2_quad)
    haskey(internal(solstep), :previous_∂C1C2∂t_quad) && copyto!(internal(solstep).previous_∂C1C2∂t_quad,C.∂C1C2∂t_quad)
    haskey(internal(solstep), :previous_∂C1C2∂x_quad) && copyto!(internal(solstep).previous_∂C1C2∂x_quad,C.∂C1C2∂x_quad)
    haskey(internal(solstep), :previous_C1C2_result) && copyto!(internal(solstep).previous_C1C2_result,C.C1C2_result)
    haskey(internal(solstep), :previous_∂C1C2∂t_result) && copyto!(internal(solstep).previous_∂C1C2∂t_result,C.∂C1C2∂t_result)
    haskey(internal(solstep), :previous_∂C1C2∂x_result) && copyto!(internal(solstep).previous_∂C1C2∂x_result,C.∂C1C2∂x_result)
end



function initialize_bcs_ics!(sol,int::PDEIntegrator{<:TrialNN_PDE_int})
    local N = int.method.initial_guess_method.N
    local quad_nodes = int.method.initial_guess_method.equispaced_quad_nodes
    local tn = sol.t - timestep(int)
    local D = int.problem.D
    local grid_matrix = int.method.grid_matrix
    local RT = int.method.RT
    local RX = int.method.RX
    local h = timestep(int)
    local x_nodes = cache(int).x_nodes

    local C1C2_result = cache(int).C1C2_result
    local ∂C1C2∂t_result = cache(int).∂C1C2∂t_result
    local ∂C1C2∂x_result = cache(int).∂C1C2∂x_result
    local C1C2_equispaced_quad_nodes = cache(int).C1C2_equispaced_quad_nodes
    local C1C2_quad = cache(int).C1C2_quad
    local ∂C1C2∂t_quad = cache(int).∂C1C2∂t_quad
    local ∂C1C2∂x_quad = cache(int).∂C1C2∂x_quad

    for i in 1:N
        C1C2_equispaced_quad_nodes[i] = C1C2(quad_nodes[:,i], tn, int,sol)
    end

    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                C1C2_quad[d, rt, rx] = C1C2(grid_matrix[rt, rx],tn,int,sol)
                ∂C1C2∂t_quad[d, rt, rx] = ∂C1C2∂t(grid_matrix[rt, rx],tn,int,sol) / h
                ∂C1C2∂x_quad[d, rt, rx] = ∂C1C2∂x(grid_matrix[rt, rx],tn,int,sol)
            end
        end
    end

    for d in 1:D
        for i in eachindex(x_nodes)
            C1C2_result[d, i] = C1C2(1.0,x_nodes[i],tn, int,sol)
            ∂C1C2∂t_result[d, i] = ∂C1C2∂t(1.0,x_nodes[i],tn, int,sol) / h
            ∂C1C2∂x_result[d, i] = ∂C1C2∂x(1.0,x_nodes[i],tn, int,sol)
        end
    end

    if tn != 0.0
        C1C2_equispaced_quad_nodes .+= cache(int).previous_C1C2_equispaced_quad_nodes
        C1C2_quad .+= cache(int).previous_C1C2_quad
        ∂C1C2∂t_quad .+= cache(int).previous_∂C1C2∂t_quad
        ∂C1C2∂x_quad .+= cache(int).previous_∂C1C2∂x_quad
        # @infiltrate
        C1C2_result .+= cache(int).previous_C1C2_result
        ∂C1C2∂t_result .+= cache(int).previous_∂C1C2∂t_result
        ∂C1C2∂x_result .+= cache(int).previous_∂C1C2∂x_result
    end


end

function components!(x::AbstractVector{ST}, sol, params, int::PDEIntegrator{<:TrialNN_PDE_int}) where {ST}
    local lag_params = int.problem.lagrangian_system.params
    local grid_matrix = int.method.grid_matrix
    local ∂L∂U = int.problem.lagrangian_system.functions.∂L∂U
    local ∂L∂V = int.problem.lagrangian_system.functions.∂L∂V
    local ∂L∂W = int.problem.lagrangian_system.functions.∂L∂W
    local D = int.problem.D
    local RT = int.method.RT
    local RX = int.method.RX
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local C = cache(int,ST)
    local S = int.method.basis.S
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local W1 = cache(int,ST).W1
    local bias1 = cache(int,ST).bias1
    local W2 = cache(int,ST).W2

    local h = timestep(int)
    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w
    local show_status = int.method.show_status

    local u_func = int.method.basis.u_func
    local v_func = int.method.basis.v_func
    local w_func = int.method.basis.w_func
    local ∂u∂p_func = int.method.basis.∂u∂p_func
    local ∂v∂p_func = int.method.basis.∂v∂p_func
    local ∂w∂p_func = int.method.basis.∂w∂p_func
    local tn = sol.t - timestep(int)
    local previous_W2_s = cache(int).previous_W2
    local previous_W1_s = cache(int).previous_W1
    local previous_bias1_s = cache(int).previous_bias1

    local C1C2_quad = cache(int).C1C2_quad
    local ∂C1C2∂t_quad = cache(int).∂C1C2∂t_quad
    local ∂C1C2∂x_quad = cache(int).∂C1C2∂x_quad

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                @views C.∂u∂θ_quad_values[d, i, j, :] = ∂u∂p_func(grid_matrix[i, j][1],grid_matrix[i, j][2],h,x)
                @views C.∂v∂θ_quad_values[d, i, j, :] = ∂v∂p_func(grid_matrix[i, j][1],grid_matrix[i, j][2],h,x)
                @views C.∂w∂θ_quad_values[d, i, j, :] = ∂w∂p_func(grid_matrix[i, j][1],grid_matrix[i, j][2],h,x)
            end
        end
    end

    # Unpack parameters from x into cache arrays without allocating slices.
    @views copyto!(W2, x[1:S])
    @views copyto!(view(W1, :, 1), x[S+1:2*S])
    @views copyto!(view(W1, :, 2), x[2*S+1:3*S])
    @views copyto!(bias1, x[3*S+1:4*S])

    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                C.u_quad_values[d, rt, rx] = u_func(grid_matrix[rt, rx][1],grid_matrix[rt, rx][2],h,x) + C1C2_quad[d, rt, rx]
                C.v_quad_values[d, rt, rx] = v_func(grid_matrix[rt, rx][1],grid_matrix[rt, rx][2],h,x) / h + ∂C1C2∂t_quad[d, rt, rx]
                C.w_quad_values[d, rt, rx] = w_func(grid_matrix[rt, rx][1],grid_matrix[rt, rx][2],h,x) + ∂C1C2∂x_quad[d, rt, rx]
            end
        end
    end

    # Compute ∂L/∂θ at quadrature points
    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                current_u = C.u_quad_values[d, i, j]
                current_v = C.v_quad_values[d, i, j]
                current_w = C.w_quad_values[d, i, j]

                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](current_u, current_v, current_w, lag_params)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](current_u, current_v, current_w, lag_params)
                C.∂L∂W_quad_values[d, i, j] = ∂L∂W[d](current_u, current_v, current_w, lag_params)
            end
        end
    end

    if show_status
        u_truth_mat = similar(C.u_quad_values)
        v_truth_mat = similar(C.v_quad_values)
        w_truth_mat = similar(C.w_quad_values)

        for d in 1:D
            for i in 1:RT
                for j in 1:RX
                    u_truth_mat[d, i, j] = int.problem.exact_u.(tn + h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                    v_truth_mat[d, i, j] = int.problem.exact_v.(tn + h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                    w_truth_mat[d, i, j] = int.problem.exact_w.(tn + h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                end
            end
        end

        @show maximum(abs.(C.u_quad_values .- u_truth_mat))
        @show maximum(abs.(C.v_quad_values .- v_truth_mat))
        @show maximum(abs.(C.w_quad_values .- w_truth_mat))

        ∂L∂U_truth_mat = similar(C.∂L∂U_quad_values)
        ∂L∂V_truth_mat = similar(C.∂L∂V_quad_values)
        ∂L∂W_truth_mat = similar(C.∂L∂W_quad_values)

        for d in 1:D
            for i in 1:RT
                for j in 1:RX
                    ∂L∂U_truth_mat[d, i, j] = ∂L∂U[d](u_truth_mat[d, i, j], v_truth_mat[d, i, j], w_truth_mat[d, i, j], lag_params)
                    ∂L∂V_truth_mat[d, i, j] = ∂L∂V[d](u_truth_mat[d, i, j], v_truth_mat[d, i, j], w_truth_mat[d, i, j], lag_params)
                    ∂L∂W_truth_mat[d, i, j] = ∂L∂W[d](u_truth_mat[d, i, j], v_truth_mat[d, i, j], w_truth_mat[d, i, j], lag_params)
                end
            end
        end

        @show maximum(abs.(C.∂L∂U_quad_values .- ∂L∂U_truth_mat))
        @show maximum(abs.(C.∂L∂V_quad_values .- ∂L∂V_truth_mat))
        @show maximum(abs.(C.∂L∂W_quad_values .- ∂L∂W_truth_mat))
    end
    # @infiltrate
end

post_initial_guess!(C, sol, int::PDEIntegrator{<:TrialNN_PDE_int}) = nothing

function residual!(b::Vector{ST}, sol, params, int::PDEIntegrator{<:TrialNN_PDE_int}) where {ST}
    local D = int.problem.D
    local RT = int.method.RT
    local RX = int.method.RX
    local NP = int.method.basis.NP

    local quad_b = int.method.grid_weights
    local C = cache(int,ST)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local show_status = int.method.show_status

    for d in 1:D
        for p in 1:NP
            z = zero(ST)
            for rt in 1:RT
                for rx in 1:RX
                    z +=  quad_b[rt,rx] *
                        ( x_domain * timestep(int) * C.∂L∂U_quad_values[d,rt,rx] * C.∂u∂θ_quad_values[d,rt,rx,p]
                        + x_domain                 * C.∂L∂V_quad_values[d,rt,rx] * C.∂v∂θ_quad_values[d,rt,rx,p]
                        + x_domain * timestep(int) * C.∂L∂W_quad_values[d,rt,rx] * C.∂w∂θ_quad_values[d,rt,rx,p])
                end
            end
            b[p] = -z
        end
    end
    # println("In the end of residual! function, b = ", b)
    # @infiltrate
    if show_status
        @show b
        @show norm(b)
    end
end

function update!(sol, int::PDEIntegrator{<:TrialNN_PDE_int})
    local D = int.problem.D
    local C = cache(int)
    local x = nlsolution(int)
    local S = int.method.basis.S
    local h = timestep(int)
    local W2 = C.W2
    local W1 = C.W1
    local bias1 = C.bias1
    local show_status = int.method.show_status
    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local u_func = int.method.basis.u_func
    local v_func = int.method.basis.v_func
    local w_func = int.method.basis.w_func
    local tn = sol.t - timestep(int)
    local h = timestep(int)
    local previous_W2_s = cache(int).previous_W2
    local previous_W1_s = cache(int).previous_W1
    local previous_bias1_s = cache(int).previous_bias1
    local x_nodes = cache(int).x_nodes
    local C1C2_result = cache(int).C1C2_result
    local ∂C1C2∂t_result = cache(int).∂C1C2∂t_result
    local ∂C1C2∂x_result = cache(int).∂C1C2∂x_result


    @views copyto!(W2, x[1:S])
    @views copyto!(view(W1, :, 1), x[S+1:2*S])
    @views copyto!(view(W1, :, 2), x[2*S+1:3*S])
    @views copyto!(bias1, x[3*S+1:4*S])


    # for d in 1:D
    #     for i in eachindex(x_nodes)
    #         sol.u[i] = u_trial(1.0, x_nodes[i],W2,W1,bias1,int,sol)
    #         sol.v[i] = v_trial_zygote(1.0,x_nodes[i],W2,W1,bias1,int,sol)/h
    #         sol.w[i] = w_trial_zygote(1.0,x_nodes[i],W2,W1,bias1,int,sol)
    #     end
    # end

    for d in 1:D
        for i in eachindex(x_nodes)
            sol.u[i] = u_func(1.0,x_nodes[i],h,x) + C1C2_result[d, i]
            sol.v[i] = v_func(1.0,x_nodes[i],h,x) / h + ∂C1C2∂t_result[d, i]
            sol.w[i] = w_func(1.0,x_nodes[i],h,x) + ∂C1C2∂x_result[d, i]
        end
    end


    if show_status
        ut₁_grid_truth = zeros(length(x_nodes))
        vt₁_grid_truth = zeros(length(x_nodes))
        wt₁_grid_truth = zeros(length(x_nodes))

        for i in eachindex(x_nodes)
            ut₁_grid_truth[i] = exact_u(sol.t, x_nodes[i])
            vt₁_grid_truth[i] = exact_v(sol.t, x_nodes[i])
            wt₁_grid_truth[i] = exact_w(sol.t, x_nodes[i])
        end

        @show maximum(abs.(sol.u .- ut₁_grid_truth))
        @show maximum(abs.(sol.v .- vt₁_grid_truth))
        @show maximum(abs.(sol.w .- wt₁_grid_truth))

        @show sol.u
        @show ut₁_grid_truth
    end

end

function internal_variables(method::TrialNN_PDE_int, problem::LPDEProblem)
    local S = method.basis.S
    local OGAN = method.initial_guess_method.N
    local D = 1
    local RT = method.RT
    local RX = method.RX
    local a,b = method.xspan[1],method.xspan[2]
    local xstep = problem.xstep

    W1 = zeros(S, 2)
    W2 = zeros(S)
    bias1 = zeros(S)
    C1C2_equispaced_quad_nodes = zeros(OGAN)
    C1C2_quad=zeros(D, RT, RX)
    ∂C1C2∂t_quad=zeros(D, RT, RX)
    ∂C1C2∂x_quad=zeros( D, RT, RX)

    x_nodes = collect(a:xstep:b)
    C1C2_result=zeros(D, length(x_nodes))
    ∂C1C2∂t_result=zeros(D, length(x_nodes))
    ∂C1C2∂x_result=zeros(D, length(x_nodes))

    return (previous_W1 = W1,
        previous_bias1 = bias1,
        previous_W2 = W2,
        previous_C1C2_equispaced_quad_nodes = C1C2_equispaced_quad_nodes,
        previous_C1C2_quad = C1C2_quad,
        previous_∂C1C2∂t_quad = ∂C1C2∂t_quad,
        previous_∂C1C2∂x_quad = ∂C1C2∂x_quad,
        previous_C1C2_result = C1C2_result,
        previous_∂C1C2∂t_result = ∂C1C2∂t_result,
        previous_∂C1C2∂x_result=∂C1C2∂x_result,
        )
end
