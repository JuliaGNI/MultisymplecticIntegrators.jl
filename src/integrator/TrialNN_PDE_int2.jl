struct TrialNN_int{BT<:AbstractPDEBasis,IPMT<:InitialParametersMethod} <: PDEMethod
    basis::BT

    RT::Int
    t_quad
    RX::Int
    x_quad

    N_train::Int # Collocation points inside the domain during initial parameter generation

    x_nodes # nodes for prediction 
    N_nodes::Int # Number of spatial nodes

    initial_guess_method::IPMT

    function TrialNN_int(NN; xspan=(0.0, 1.0), xstep=0.01, RT::Int=6, RX::Int=8, N_train::Int=600, initial_guess_method::IPMT=ELM()) where {IPMT}
        t_quadrature = QuadratureRules.GaussLegendreQuadrature(RT)
        x_quadrature = QuadratureRules.GaussLegendreQuadrature(RX)
        x_nodes = collect(xspan[1]:xstep:xspan[2])
        new{typeof(NN),typeof(initial_guess_method)}(NN,
            RT, t_quadrature,
            RX, x_quadrature,
            N_train,
            x_nodes, length(x_nodes),
            initial_guess_method)
    end
end

default_solver(::TrialNN_int) = Newton()

struct TrialNN_intCache{ST,RT,RX,D,S,N} <: PDEIntegratorCache{ST,D}
    """
    RT = number of quadrature points in time
    RX = number of quadrature points in space
    D = dimension of output of u,v,w, i.e. scaler value function (D = 1) or vector function
    S = number of parameters in the expression
    """
    x::Vector{ST}

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

    ∂u∂θ_quad_values::Array{ST}
    ∂v∂θ_quad_values::Array{ST}
    ∂w∂θ_quad_values::Array{ST}

    basis_nn_ps
    function TrialNN_intCache{ST,RT,RX,D,S}() where {ST,RT,RX,D,S}
        x = zeros(ST, S) # in ELM, x is just the output layer parameters

        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        ∂u∂θ_quad_values = zeros(ST, D, RT, RX, S)
        ∂v∂θ_quad_values = zeros(ST, D, RT, RX, S)
        ∂w∂θ_quad_values = zeros(ST, D, RT, RX, S)

        basis_nn_ps = (L1=(W=zeros(ST, S, 2), b=zeros(ST, S)), )

        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ∂u∂θ_quad_values, ∂v∂θ_quad_values, ∂w∂θ_quad_values,
            basis_nn_ps,
        )
    end
end

nlsolution(cache::TrialNN_intCache) = cache.x

function Cache{ST}(problem::PDEProblem, int::TrialNN_int; kwargs...) where {ST}
    TrialNN_intCache{ST,int.RT,int.RX,problem.D,int.basis.S}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::PDEProblem, int::TrialNN_int) = TrialNN_intCache{ST,int.RT,int.RX,problem.D,int.basis.S}

@inline function Base.getindex(c::TrialNN_intCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function T1NN_BNN(t, x, int)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local a, b = int.problem.xspan[1], int.problem.xspan[2]
    local BNN = int.method.basis.basis_network
    local BNN_ps = cache(int).basis_nn_ps

    return (b - x) / x_domain * BNN([t, a],BNN_ps) +
           (x - a) / x_domain * BNN([t, b],BNN_ps) +
           (1 - t) * BNN([0.0, x],BNN_ps)
end

function T2NN_BNN(t, x, int)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local a, b = int.problem.xspan[1], int.problem.xspan[2]
    local BNN = int.method.basis.basis_network
    local BNN_ps = cache(int).basis_nn_ps

    return (b - x) / x_domain * (1 - t) * BNN([0.0, a],BNN_ps) +
           (x - a) / x_domain * (1 - t) * BNN([0.0, b],BNN_ps)
end

function init_function_BNN(t, x, int, sol)
    local ic_fun = int.problem.ics_function
    local current_step = sol.current_step

    if current_step == 1
        return ic_fun(x).u
    else
        return sol.internal.xx[current_step-1, :]' * BNN([t, x])
    end
end

function C1_BNN(t, x, int, sol)
    local xspan = int.problem.xspan
    local a, b = xspan[1], xspan[2]
    local x_domain = b - a
    local bc_fun = int.problem.bcs_function

    return (b - x) / x_domain * bc_fun(t, xspan).bc₀.u +
           (x - a) / x_domain * bc_fun(t, xspan).bc₁.u +
           (1 - t) * init_function(t, x, int, sol)
end

function C2_BNN(t, x, int, sol)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local a, b = int.problem.xspan[1], int.problem.xspan[2]
    local bc_fun = int.problem.bcs_function
    local xspan = int.problem.xspan
    local h = timestep(int)
    local tn = (sol.current_step - 1) * h

    return (b - x) / x_domain * (1 - t) * bc_fun(tn, xspan).bc₀.u +
           (x - a) / x_domain * (1 - t) * bc_fun(tn, xspan).bc₁.u
end


function prior_initial_guess!(C, sol, int::PDEIntegrator{<:TrialNN_int})
    local N_train = int.method.N_train
    local a, b = int.problem.xspan[1], int.problem.xspan[2]
    local exact_u = int.problem.exact_u
    local h = timestep(int)
    local NN_width = int.method.basis.NN_width

    # Define a temporary network with same activation function and NN width
    NN = NeuralNetwork(Chain(Dense(d, NN_width, activation_function),Dense(NN_width,1,identity,use_bias = false)))

    # Prepare random training data.
    collocation_points = rand(2, N_train)
    collocation_points[2,:] = a .+ (b - a) * collocation_points[2,:]

    rhs = exact_u(h .* collocation_points[1, :], collocation_points[2, :]) .- C1(collocation_points[1, :], collocation_points[2, :], int, sol) .+ C2(collocation_points[1, :], collocation_points[2, :], int, sol)
    A = BNN(collocation_points) .- T1NN.(collocation_points[1, :], collocation_points[2, :], int) .+ T2NN.(collocation_points[1, :], collocation_points[2, :], int)
end



function components!(x::AbstractVector{ST}, sol, int::PDEIntegrator{<:TrialNN_PDE_int}) where {ST}
    local C = cache(int, ST)
    C.basis_nn_ps.L2.W[:] = x[:]

    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                C.u_quad_values[d, rt, rx] =
                    C.v_quad_values[d, rt, rx] = v_trial(t_quad_nodes[rt], x_quad_nodes[rx], tn, C.basis_nn_ps, int, sol)
                C.w_quad_values[d, rt, rx] = w_trial(t_quad_nodes[rt], x_quad_nodes[rx], tn, C.basis_nn_ps, int, sol)
            end
        end
    end

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], problem_params)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], problem_params)
                C.∂L∂W_quad_values[d, i, j] = ∂L∂W[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], problem_params)
            end
        end
    end
end

function internal_variables(int::PDEIntegrator{<:TrialNN_PDE_int}, problem::PDEProblem)
    ntime = Int((problem.tspan[2] - problem.tspan[1]) / problem.tstep)
    NP = int.method.basis.NP
    xx = zeros(Float64, ntime, NP) # Store the solution at all time steps
    return (xx=xx,
    )
end

