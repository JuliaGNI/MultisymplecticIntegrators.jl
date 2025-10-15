struct NN_PDE_Integrator{T,MVT,LT,BT<:AbstractPDEBasis, IPMT<:InitialParametersMethod} <: PDEMethod
    basis::BT
    time_quadrature
    RT::Int # Number of quadrature points in time

    spatial_quadrature
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions
    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights
    
    k_μ::Int
    μ₀_t::MVT
    μ₁_t::MVT

    k_λ₀_x::Int
    λ₀_x::LT

    mλ₀_x # λ₀_x evaluated at quadrature points
    mμ_t
    nepochs::Int
    initial_guess_method::IPMT # :LSGD or :GroundTruth
    params_turbulance::Float64 # a small value to add to the parameters to avoid zeros in the system
    function NN_PDE_Integrator(basis; RT::Int = 6,RX::Int = 8,xspan::Tuple = (0.,1.0),tstep::T = 1.0, k_μ::Int = 4,k_λ₀_x::Int = 4,
        μ::Symbol = :BSplineDirichlet,λ::Symbol= :BSplineDirichlet,nepochs = 1000,initial_guess_method::IPMT=LSGD(),params_turbulance = 1e-7) where {T, IPMT}
        
        if RT ==128 
            t_quadrature = GaussQuadrature128()
        elseif RT == 64
            t_quadrature = GaussQuadrature64()
        else
            t_quadrature = QuadratureRules.GaussLegendreQuadrature(RT)
        end

        if RX == 128
            x_quadrature = GaussQuadrature128()
        elseif RX == 64
            x_quadrature = GaussQuadrature64()
        else
            x_quadrature = QuadratureRules.GaussLegendreQuadrature(RX)
        end

        dimensions = [RT,RX]  
        grid_matrix, grid_weights = construct_quadrature_grid(dimensions)

        μ₀_t = Lagrangian_multiplier(μ,k_μ,tstep .* t_quadrature.nodes)
        μ₁_t = Lagrangian_multiplier(μ,k_μ,tstep .* t_quadrature.nodes)
        λ₀_x = Lagrangian_multiplier(λ,k_λ₀_x,xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes)

        mλ₀_x = zeros(RX, RX)
        mμ_t = zeros(RT, RT)

        for i in 1:RX
            mλ₀_x[i,:] = λ₀_x.b[i].(xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes)
        end
    
        for i in 1:RT
            mμ_t[i,:] = μ₀_t.b[i].(tstep .* t_quadrature.nodes)
        end

        new{T,typeof(μ₀_t),typeof(λ₀_x),typeof(basis),typeof(initial_guess_method)}(basis, t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            k_μ, μ₀_t, μ₁_t,
            k_λ₀_x,λ₀_x, #λ₁_x,
            mλ₀_x, mμ_t,
            nepochs,
            initial_guess_method,params_turbulance)
    end 
end

default_solver(::NN_PDE_Integrator) = Newton()

struct NN_PDE_IntegratorCache{ST,RT,RX,D,NP} <: PDEIntegratorCache{ST,D}
    """
    RT = number of quadrature points in time
    RX = number of quadrature points in space
    D = dimension of output of u,v,w, i.e. scaler value function (D = 1) or vector function
    DX = dimension of input of u,v,w, i.e. number of spatial dimensions
    NP = number of parameters in the expression
    """
    x::Vector{ST}

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

    ∂u∂P_quad_values
    ∂v∂P_quad_values
    ∂w∂P_quad_values

    λ₀_x_coes::Matrix{ST}

    μ₀_t_coes::Matrix{ST}
    μ₁_t_coes::Matrix{ST}

    λ₀_quad_values::Matrix{ST} 
    μ₀_quad_values::Matrix{ST}
    μ₁_quad_values::Matrix{ST} 

    ∂u∂P_t₀_quad_values
    ∂u∂P_t₁_quad_values
    ∂u∂P_x₀_quad_values
    ∂u∂P_x₁_quad_values

    ut₀_quad_values::Matrix{ST} # bottom boundary, i.e. t = 0
    ut₁_quad_values::Matrix{ST} # top boundary, i.e. t = T
    vt₀_quad_values::Matrix{ST}
    vt₁_quad_values::Matrix{ST}
    wt₀_quad_values::Matrix{ST}
    wt₁_quad_values::Matrix{ST}

    ux₀_quad_values::Matrix{ST} # left boundary, i.e. x = 0
    ux₁_quad_values::Matrix{ST} # right boundary, i.e. x = L
    vx₀_quad_values::Matrix{ST}
    vx₁_quad_values::Matrix{ST}
    wx₀_quad_values::Matrix{ST}
    wx₁_quad_values::Matrix{ST}

    init_condition_t₀::Matrix{ST}
    boundary_condition_x₀::Matrix{ST}
    boundary_condition_x₁::Matrix{ST}

    sol_params
    function NN_PDE_IntegratorCache{ST,RT,RX,D,NP}(network_arch) where {ST,RT,RX,D,NP}
        # x = zeros(ST, NP + 2 * D * RX + 2* D * DX * RT ) # TODO: how to deal with RX being a vector/
        x = zeros(ST,NP + D * RX +2* D * RT) # params, λ₀_x_coes,μ₀_t_coes,μ₁_t_coes
        # TODO:consider when DX is a vector
        
        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        ∂u∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,[NP])
        ∂v∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,[NP])
        ∂w∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,[NP])
        
        λ₀_x_coes = zeros(ST, D, RX)

        μ₀_t_coes = zeros(ST, D, RT)
        μ₁_t_coes = zeros(ST, D, RT)

        λ₀_quad_values = zeros(ST, D, RX) 
        μ₀_quad_values = zeros(ST, D, RT) 
        μ₁_quad_values = zeros(ST, D, RT) 

        ∂u∂P_t₀_quad_values = create_boundary_derivative_vector(ST, D, RX,[NP]) 
        ∂u∂P_t₁_quad_values = create_boundary_derivative_vector(ST, D, RX,[NP]) 
        ∂u∂P_x₀_quad_values = create_boundary_derivative_vector(ST, D, RT,[NP]) 
        ∂u∂P_x₁_quad_values = create_boundary_derivative_vector(ST, D, RT,[NP]) 

        ut₀_quad_values = zeros(ST,D,RX) # bottom boundary, i.e. t = 0
        ut₁_quad_values = zeros(ST,D,RX) # top boundary, i.e. t = T
        vt₀_quad_values = zeros(ST,D,RX)
        vt₁_quad_values = zeros(ST,D,RX)
        wt₀_quad_values = zeros(ST,D,RX)
        wt₁_quad_values = zeros(ST,D,RX)

        ux₀_quad_values = zeros(ST,D,RT) # left boundary, i.e. x = 0
        ux₁_quad_values = zeros(ST,D,RT) # right boundary, i.e. x = L
        vx₀_quad_values = zeros(ST,D,RT)
        vx₁_quad_values = zeros(ST,D,RT)
        wx₀_quad_values = zeros(ST,D,RT)
        wx₁_quad_values = zeros(ST,D,RT)

        init_condition_t₀ = zeros(ST, D, RX)

        boundary_condition_x₀ = zeros(ST, D, RT)
        boundary_condition_x₁ = zeros(ST, D, RT)

        sol_params = network_cache_create(network_arch,ST)

        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ∂u∂P_quad_values, ∂v∂P_quad_values, ∂w∂P_quad_values,
            λ₀_x_coes, μ₀_t_coes,μ₁_t_coes, 
            λ₀_quad_values, μ₀_quad_values, μ₁_quad_values,
            ∂u∂P_t₀_quad_values, ∂u∂P_t₁_quad_values, ∂u∂P_x₀_quad_values, ∂u∂P_x₁_quad_values,
            ut₀_quad_values, ut₁_quad_values,vt₀_quad_values, vt₁_quad_values,wt₀_quad_values, wt₁_quad_values,
            ux₀_quad_values, ux₁_quad_values,vx₀_quad_values, vx₁_quad_values,wx₀_quad_values, wx₁_quad_values,
            init_condition_t₀, 
            boundary_condition_x₀, boundary_condition_x₁,
            sol_params)
    end
end

nlsolution(cache::NN_PDE_IntegratorCache) = cache.x

function Cache{ST}(problem::PDEProblem, int::NN_PDE_Integrator; kwargs...) where {ST}
    NN_PDE_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.NP}(int.basis.network_arch; kwargs...)
end

#{ST,RT,RX,D,NP}(NP) where {ST,RT,RX,D,NP}
@inline GeometricIntegrators.Integrators.CacheType(ST, problem::PDEProblem, int::NN_PDE_Integrator) = NN_PDE_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.NP}

@inline function Base.getindex(c::NN_PDE_IntegratorCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

# function prior_initial_guess!(C,sol,int::PDEIntegrator{<:NN_PDE_Integrator{T,MVT,LT,BT,IPMT}}) where {T,MVT,LT,BT,IPMT<:ELM_LS}

#     function nlls!(du, u, int::PDEIntegrator{<:NN_PDE_Integrator})
#         local xspan = int.problem.xspan
#         local c = int.problem.params.c
#         local tn = (sol.current_step - 1) * timestep(int)
#         local N_in = 2000

#         tx_in = rand(Random.seed!(1),2,N_in)
#         tx_in[2,:] .= xspan[1] .+ (xspan[2] - xspan[1]) * tx_in[2,:]

#         for i in 1:N_in
#             du[i] = v_trial(tx_in[1,i], tx_in[2,i], tn, u) + c * w_trial(tx_in[1,i], tx_in[2,i], tn, u)
#         end
#     end

#     if C.done_initial_guess[1] == 0
#         u0 = zeros(1,NP)
#         prob = NonlinearLeastSquaresProblem(
#         NonlinearFunction(nlls!, resid_prototype = zeros(N_in)), u0, int)
#         println("Starting initial guess computation ...")
#         t1 = time()
#         u_sol = solve(prob,maxtime = 60,abstol = 1e-12, reltol = 1e-12).u
#         (x == cache(int).x && eltype(x) == eltype(cache(int).x)) ? x[:] = u_sol : nothing
#         println("Time for initial guess: ", time() - t1)
#         print("initial guess parameters: ", x, "\n")
#         C.done_initial_guess[1] = 1
#     end
# end


function prior_initial_guess!(C,sol,int::PDEIntegrator{<:NN_PDE_Integrator{T,MVT,LT,BT,IPMT}}) where {T,MVT,LT,BT,IPMT<:LSGD}
    local NP = int.method.basis.NP
    local RT = int.method.RT
    local RX = int.method.RX
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local NN = int.method.basis.network_arch
    local PNN = int.method.basis.u
    local nepochs = int.method.nepochs
    local optim_mode = int.method.basis.optim_mode

    network_inputs,_ = construct_quadrature_grid_with_boundary([RT,RX])
    labels = int.problem.exact_u.(sol.t .- timestep(int) .+ timestep(int) .* network_inputs[1,:],int.problem.xspan[1] .+ x_domain .* network_inputs[2,:])
    labels = reshape(labels,1,:) # labels should be a matrix with one row and multiple columns

    # initialize the parameters and train with LSGD
    for (name, layer) in zip(keys(PNN.params), values(PNN.params))
        in_size = size(layer.W, 2)
        out_size = size(layer.W, 1)
        if hasfield(typeof(layer), :b)
            layer.W[:], layer.b[:] = box_init_plain(in_size, out_size)
        else
            # For layers without bias (e.g., output), just regenerate W
            layer.W[:], _ = box_init_plain(in_size, out_size)
        end
    end

    tem_ps = PNN.params[keys(PNN.params)[1:end-1]]
    opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.GradientOptimizer(.0005), tem_ps)
    err = 0
    λ = GeometricMachineLearning.GlobalSection(tem_ps)

    for ep in 1:nepochs
        Φ = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)(network_inputs,tem_ps)
        # Φ = NN(network_inputs, PNN.params)
        # PNN.params.L3.W[:] = labels/Φ
        PNN.params[keys(PNN.params)[end]].W[:] = (Φ' \ labels')'
        gs = Zygote.gradient(p -> lsgd_loss(network_inputs,labels,NN,p),PNN.params)[1]
        tem_ps = PNN.params[keys(PNN.params)[1:end-1]]
        tem_gs = gs[keys(gs)[1:end-1]]
        GeometricMachineLearning.optimization_step!(opt,λ, tem_ps, tem_gs)
        err = lsgd_loss(network_inputs,labels,NN,PNN.params)
        if err < 5e-8
            print("\n final loss: $err by $ep epochs")
            break
        elseif ep == nepochs
            print("\n final loss: $err by $ep epochs")
        end
    end

    # copy the parameters to the cache
    if optim_mode == :Partially
        C.x[1:NP] = PNN.params[keys(PNN.params)[end]].W[:]
    elseif optim_mode == :Fully
        C.x[1:NP]= flatten_params(PNN.params)
    end

end

# function prior_initial_guess!(C,sol,int::PDEIntegrator{<:NN_PDE_Integrator{T,MVT,LT,BT,IPMT}}) where {T,MVT,LT,BT,IPMT<:GroundTruth}
#     local NP = int.method.basis.NP
#     local PNN = int.method.basis.u
#     local optim_mode = int.method.basis.optim_mode
#     local params_turbulance = int.method.params_turbulance

#     PNN.params.L1.W[:,2] .= 1.0 
#     PNN.params.L1.W[:,1] .= - 0.2
#     PNN.params.L1.b[:] = [-0.0000, -0.2930, -0.5664, -0.1328, -0.7207, -0.3965, -0.4893, -0.6426]
#     PNN.params.L2.W[:] = [0.3275,   52.7569,   -6.1713,   -1.7977,    6.4468, -153.8942,137.0191,  -34.2571]

#     # copy the parameters to the cache
#     if optim_mode == :Partially
#         PNN.params[keys(PNN.params)[end]].W[:] = PNN.params[keys(PNN.params)[end]].W[:] .+ params_turbulance .* randn(Random.seed!(1),NP, 1)
#         C.x[1:NP] = PNN.params[keys(PNN.params)[end]].W[:]
#     elseif optim_mode == :Fully
#         # Purtube the parameters a little bit to avoid zeros in the syste
#         for (name, layer) in zip(keys(PNN.params), values(PNN.params))
#             if hasfield(typeof(layer), :b)
#                 layer.W[:] = layer.W[:] .+ params_turbulance .* randn(Random.seed!(1), size(layer.W[:]))
#                 layer.b[:] = layer.b[:] .+ params_turbulance .* randn(Random.seed!(1), size(layer.b[:]))
#             else
#                 # For layers without bias (e.g., output), just regenerate W
#                 layer.W[:] = layer.W[:] .+ params_turbulance .* randn(Random.seed!(2),size(layer.W[:]))
#             end
#         end
#         C.x[1:NP]= flatten_params(PNN.params)
#     end

# end

function prior_initial_guess!(C,sol,int::PDEIntegrator{<:NN_PDE_Integrator{T,MVT,LT,BT,IPMT}}) where {T,MVT,LT,BT,IPMT<:GroundTruth}
    local NP = int.method.basis.NP
    local PNN = int.method.basis.u
    local optim_mode = int.method.basis.optim_mode
    local params_turbulance = int.method.params_turbulance

    PNN.params.L1.W[:,2] .= [-1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0]
    PNN.params.L1.W[:,1] .= - 0.2 * [-1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0]
    PNN.params.L1.b[:] = [ 0.9995, -0.0000,  0.6479,  0.3423, -0.5156, -0.7178,  0.2222,  0.7964,
        -0.4072,  0.4653, -0.2852,  0.5786,  0.1450, -0.3730, -0.4873, -0.6123,
        -0.2559,  0.4321,  0.3091,  0.3911]
    PNN.params.L2.W[:] = [-5.5742e+00,  1.1024e+01, -5.3139e+00, -4.0719e+00,  6.0699e+01,
         -7.4047e-01, -9.1054e-01, -1.7195e-01, -1.5121e+02,  4.6744e+01,
          2.2065e+01,  2.3221e+01,  1.1691e-01, -2.3493e+01,  1.2785e+02,
         -3.6581e+01, -4.0092e+00, -4.0954e+01, -3.2466e+01,  8.3772e+00]

    # copy the parameters to the cache
    if optim_mode == :Partially
        PNN.params[keys(PNN.params)[end]].W[:] = PNN.params[keys(PNN.params)[end]].W[:] .+ params_turbulance .* randn(Random.seed!(1),NP, 1)
        C.x[1:NP] = PNN.params[keys(PNN.params)[end]].W[:]
    elseif optim_mode == :Fully
        # Purtube the parameters a little bit to avoid zeros in the syste
        for (name, layer) in zip(keys(PNN.params), values(PNN.params))
            if hasfield(typeof(layer), :b)
                layer.W[:] = layer.W[:] .+ params_turbulance .* randn(size(layer.W[:]))
                layer.b[:] = layer.b[:] .+ params_turbulance .* randn(size(layer.b[:]))
            else
                # For layers without bias (e.g., output), just regenerate W
                layer.W[:] = layer.W[:] .+ params_turbulance .* randn(size(layer.W[:]))
            end
        end
        C.x[1:NP]= flatten_params(PNN.params)
    end

end
function initialize_bcs_ics!(sol,int::PDEIntegrator{<:NN_PDE_Integrator})
    local C = cache(int)
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local u = [int.method.basis.u]
    local D = int.problem.D 
    local RT = int.method.RT
    local ic_fun = int.problem.ics_function
    local bc_fun = int.problem.bcs_function
    local current_step = sol.current_step
    local NP = int.method.basis.NP
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]

    for d in 1:D
        # println("update initial condition, current time = ", sol.t, "the initial condition is at time = ", sol.t - timestep(int))

        if current_step ==1 
            C.init_condition_t₀[d,:] .= ic_fun(xspan[1] .+ x_domain .* x_quad_nodes).u
        else
            # println("sol.internal.x[current_step-1][1:NP] = " , sol.internal.x[current_step-1][1:NP])
            sol_params = NeuralNetworkParameters(reconstruct_params(sol.internal.x[current_step-1][1:NP], u[d].params))
            for i in eachindex(C.init_condition_t₀[d,:])
                C.init_condition_t₀[d,i] = u[d]([1.0,xspan[1] + x_domain * x_quad_nodes[i]], sol_params)[1]
            end
            # println("initial condition = " , C.init_condition_t₀[d,:])
        end

        for i in 1:RT
            C.boundary_condition_x₀[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₀.u
            C.boundary_condition_x₁[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₁.u
        end
        # println("left boundary condition = " , C.boundary_condition_x₀[d,:])
    end

end

function post_initial_guess!(C,sol_struct,int::PDEIntegrator{<:NN_PDE_Integrator},int_method::NN_PDE_Integrator{T,MVT,LT,BT, IPMT}) where {T,MVT<:BSplineDirichlet{T},LT<:BSplineDirichlet{T},BT,IPMT}
    local NP = int_method.basis.NP
    local RT = int_method.RT
    local RX = int_method.RX
    local D = int.problem.D 
    local lag_sys = int.problem.lagrangian_system.functions
    local params = int.problem.lagrangian_system.params
    local xspan = int.problem.xspan
    local x_quad_nodes = int_method.spatial_quadrature.nodes
    local k_λ₀_x = int_method.k_λ₀_x
    local t_quad_nodes = int_method.time_quadrature.nodes
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    # for d in 1:D
    #     for i in 1:RX
    #         C.x[NP + (d - 1) * RX + i] = lag_sys.∂L∂V[d](C.ut₀_quad_values[d,i], C.vt₀_quad_values[d,i], C.wt₀_quad_values[d,i], params)
    #     end

    #     for i in 1:RT
    #         C.x[NP+ D * RX+(d-1)*RT+i] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,i], C.vx₀_quad_values[d,i], C.wx₀_quad_values[d,i], params)
    #         C.x[NP+ D * RX+ D * RT + (d-1)*RT+i] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,i], C.vx₁_quad_values[d,i], C.wx₁_quad_values[d,i], params)
    #     end
    # end

    for d in 1:D
        tem_t₀_∂L∂V = zeros(RX)
        for i in 1:RX
            tem_t₀_∂L∂V[i] = lag_sys.∂L∂V[d](C.ut₀_quad_values[d,i], C.vt₀_quad_values[d,i], C.wt₀_quad_values[d,i], params)
        end
    
        tem_x = xspan[1] .+ x_domain .* x_quad_nodes
        λ₀_x = interpolate(tem_x, tem_t₀_∂L∂V, BSplineOrder(k_λ₀_x))
        for i in 1:RX
            C.x[NP + (d - 1) * RX + i] = λ₀_x.spline.coefs[i]
        end

        tem_x₀_∂L∂W = zeros(RT)
        for i in 1:RT
            tem_x₀_∂L∂W[i] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,i], C.vx₀_quad_values[d,i], C.wx₀_quad_values[d,i], params)
        end

        tem_t = sol_struct.t .- timestep(int) .+ timestep(int) .* t_quad_nodes
        μ₀_t = interpolate(tem_t, tem_x₀_∂L∂W, BSplineOrder(int_method.k_μ))
        for i in 1:RT
            C.x[NP + D * RX + (d - 1) * RT + i] = μ₀_t.spline.coefs[i]
        end

        tem_x₁_∂L∂W = zeros(RT)
        for i in 1:RT
            tem_x₁_∂L∂W[i] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,i], C.vx₁_quad_values[d,i], C.wx₁_quad_values[d,i], params)
        end

        μ₁_t = interpolate(tem_t, tem_x₁_∂L∂W, BSplineOrder(int_method.k_μ))
        for i in 1:RT
            C.x[NP+ D * RX + D * RT + (d-1)*RT+i] = μ₁_t.spline.coefs[i]
        end

    end
end

function post_initial_guess!(C,sol_struct,int::PDEIntegrator{<:NN_PDE_Integrator},int_method::NN_PDE_Integrator{T,MVT,LT,BT, IPMT}) where {T,MVT<:Lagrange,LT<:Lagrange,BT,IPMT}
    local NP = int_method.basis.NP
    local RT = int_method.RT
    local RX = int_method.RX
    local D = int.problem.D 
    local lag_sys = int.problem.lagrangian_system.functions
    local params = int.problem.lagrangian_system.params

    # for d in 1:D
    #     C.x[NP+1:NP+RX] = lag_sys.∂L∂V[d].(C.ut₀_quad_values[d,:], C.vt₀_quad_values[d,:], C.wt₀_quad_values[d,:], params)
    #     C.x[NP+RX+1:NP+RX+RT] = lag_sys.∂L∂W[d].(C.ux₀_quad_values[d,:], C.vx₀_quad_values[d,:], C.wx₀_quad_values[d,:], params)
    #     C.x[NP+RX+RT+1:NP+RX+2*RT] = lag_sys.∂L∂W[d].(C.ux₁_quad_values[d,:], C.vx₁_quad_values[d,:], C.wx₁_quad_values[d,:], params)
    # end

    for d in 1:D
        for i in 1:RX
            C.x[NP + (d - 1) * RX + i] = lag_sys.∂L∂V[d](C.ut₀_quad_values[d,i], C.vt₀_quad_values[d,i], C.wt₀_quad_values[d,i], params)
        end

        for i in 1:RT
            C.x[NP+ D * RX+(d-1)*RT+i] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,i], C.vx₀_quad_values[d,i], C.wx₀_quad_values[d,i], params)
            C.x[NP+ D * RX+ D * RT + (d-1)*RT+i] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,i], C.vx₁_quad_values[d,i], C.wx₁_quad_values[d,i], params)
        end
    end
end

function components!(x::AbstractVector{ST}, sol, int::PDEIntegrator{<:NN_PDE_Integrator}) where {ST}
    local C = cache(int,ST)

    local NP = int.method.basis.NP
    local RT = int.method.RT
    local RX = int.method.RX
    local D = int.problem.D

    local ∂L∂U = int.problem.lagrangian_system.functions.∂L∂U
    local ∂L∂V = int.problem.lagrangian_system.functions.∂L∂V
    local ∂L∂W = int.problem.lagrangian_system.functions.∂L∂W

    local u = [int.method.basis.u]
    local v = [int.method.basis.v]
    local w = [int.method.basis.w]

    local ∂u∂P = int.method.basis.∂u∂P
    local ∂v∂P = int.method.basis.∂v∂P
    local ∂w∂P = int.method.basis.∂w∂P

    local grid_matrix = int.method.grid_matrix
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]

    local params = int.problem.lagrangian_system.params
    local mλ₀_x = int.method.mλ₀_x
    local mμ_t = int.method.mμ_t
    local optim_mode = int.method.basis.optim_mode
    last_layer = keys(u[1].params)[end]

    #copy part of x into the network parameter
    if optim_mode == :Fully
        sol_params = NeuralNetworkParameters(reconstruct_params(x[1:NP], u[1].params))
    elseif optim_mode == :Partially
        for (name, layer) in zip(keys(u[1].params), values(u[1].params))
            if hasfield(typeof(layer), :b)
                C.sol_params[name].W[:] = layer.W[:] 
                C.sol_params[name].b[:] = layer.b[:]
            else
                # For layers without bias (e.g., output), just regenerate W
                C.sol_params[name].W[:] = layer.W[:]
            end
        end

        C.sol_params[last_layer].W[:] = x[1:NP]
    end
    # interior values at quadrature points
    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.u_quad_values[d, i, j] = (u[d])([grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2]],C.sol_params)[1]
                C.v_quad_values[d, i, j] = (v[d])([grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2]],C.sol_params)[1]
                C.w_quad_values[d, i, j] = (w[d])([grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2]],C.sol_params)[1]
            end
        end
    end

    if optim_mode == :Fully
        for d in 1:D
            for i in 1:RT
                for j in 1:RX#TODO what if RX is a Vector
                    C.∂u∂P_quad_values[d][i, j, :] = flatten_params(∂u∂P([grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2]],C.sol_params))
                    C.∂v∂P_quad_values[d][i, j, :] = flatten_params(∂v∂P([grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2]],C.sol_params))
                    C.∂w∂P_quad_values[d][i, j, :] = flatten_params(∂w∂P([grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2]],C.sol_params))
                end
            end
        
            for rx in 1:RX
                C.∂u∂P_t₀_quad_values[d][rx,:] = flatten_params(∂u∂P([0.0, xspan[1] + x_domain* x_quad_nodes[rx]],C.sol_params))
                C.∂u∂P_t₁_quad_values[d][rx,:] = flatten_params(∂u∂P([1.0, xspan[1] + x_domain* x_quad_nodes[rx]],C.sol_params))
            end
            for rt in 1:RT
                C.∂u∂P_x₀_quad_values[d][rt,:] = flatten_params(∂u∂P([t_quad_nodes[rt], xspan[1]],C.sol_params))
                C.∂u∂P_x₁_quad_values[d][rt,:] = flatten_params(∂u∂P([t_quad_nodes[rt], xspan[2]],C.sol_params))
            end
        end
    elseif optim_mode == :Partially
        for d in 1:D
            for i in 1:RT
                for j in 1:RX#TODO what if RX is a Vector
                    C.∂u∂P_quad_values[d][i, j, :] = ∂u∂P([grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2]],C.sol_params)[last_layer].W[:]
                    C.∂v∂P_quad_values[d][i, j, :] = ∂v∂P([grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2]],C.sol_params)[last_layer].W[:]
                    C.∂w∂P_quad_values[d][i, j, :] = ∂w∂P([grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2]],C.sol_params)[last_layer].W[:]
                end
            end
        
            for rx in 1:RX
                C.∂u∂P_t₀_quad_values[d][rx,:] = ∂u∂P([0.0, xspan[1] + x_domain* x_quad_nodes[rx]],C.sol_params)[last_layer].W[:]
                C.∂u∂P_t₁_quad_values[d][rx,:] = ∂u∂P([1.0, xspan[1] + x_domain* x_quad_nodes[rx]],C.sol_params)[last_layer].W[:]
            end
            for rt in 1:RT
                C.∂u∂P_x₀_quad_values[d][rt,:] = ∂u∂P([t_quad_nodes[rt], xspan[1]],C.sol_params)[last_layer].W[:]
                C.∂u∂P_x₁_quad_values[d][rt,:] = ∂u∂P([t_quad_nodes[rt], xspan[2]],C.sol_params)[last_layer].W[:]
            end
        end
    end



    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], params)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], params)
                C.∂L∂W_quad_values[d, i, j] = ∂L∂W[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], params)
            end
        end 
    end

    # boundary values at quadrature points
    for d in 1:D
        for j in 1:RX
            C.ut₀_quad_values[d,j] = u[d]([0.0 ,xspan[1] + x_domain* x_quad_nodes[j]],C.sol_params)[1] # bottom 
            C.ut₁_quad_values[d,j] = u[d]([1.0 ,xspan[1] + x_domain* x_quad_nodes[j]],C.sol_params)[1] # top

            C.vt₀_quad_values[d,j] = v[d]([0.0 ,xspan[1] + x_domain* x_quad_nodes[j]],C.sol_params)[1]
            C.vt₁_quad_values[d,j] = v[d]([1.0 ,xspan[1] + x_domain* x_quad_nodes[j]],C.sol_params)[1]

            C.wt₀_quad_values[d,j] = w[d]([0.0 ,xspan[1] + x_domain* x_quad_nodes[j]],C.sol_params)[1]
            C.wt₁_quad_values[d,j] = w[d]([1.0 ,xspan[1] + x_domain* x_quad_nodes[j]],C.sol_params)[1]
        end

        for i in 1:RT
            C.ux₀_quad_values[d,i] = u[d]([t_quad_nodes[i],xspan[1]],C.sol_params)[1]
            C.ux₁_quad_values[d,i] = u[d]([t_quad_nodes[i],xspan[2]],C.sol_params)[1]
            C.vx₀_quad_values[d,i] = v[d]([t_quad_nodes[i],xspan[1]],C.sol_params)[1]
            C.vx₁_quad_values[d,i] = v[d]([t_quad_nodes[i],xspan[2]],C.sol_params)[1]
            C.wx₀_quad_values[d,i] = w[d]([t_quad_nodes[i],xspan[1]],C.sol_params)[1]
            C.wx₁_quad_values[d,i] = w[d]([t_quad_nodes[i],xspan[2]],C.sol_params)[1]
        end

    end

    (x == cache(int).x && eltype(x) == eltype(cache(int).x)) ? post_initial_guess!(cache(int),sol,int,int.method) : nothing

    for d in 1:D
        C.λ₀_x_coes[d,:] = x[NP+1:NP+RX]
        C.μ₀_t_coes[d,:] = x[NP+RX+1:NP+RX+RT]
        C.μ₁_t_coes[d,:] = x[NP+RX+RT+1:NP+RX+2*RT]
    end

    for d in 1:D
        for rx in 1:RX
            C.λ₀_quad_values[d,rx] = sum([C.λ₀_x_coes[d,i] * mλ₀_x[i,rx]  for i in 1:RX])
        end

        for rt in 1:RT
            C.μ₀_quad_values[d,rt] = sum([C.μ₀_t_coes[d,i]*mμ_t[i,rt] for i in 1:RT])
            C.μ₁_quad_values[d,rt] = sum([C.μ₁_t_coes[d,i]*mμ_t[i,rt] for i in 1:RT])
        end
    end
end

function residual!(b::Vector{ST}, sol, int::PDEIntegrator{IT}) where {ST,T, MVT <: BSplineDirichlet,LT  <: BSplineDirichlet,BT,IT <: NN_PDE_Integrator{T, MVT, LT, BT}}
    local D = int.problem.D 
    local D = int.problem.D 
    local RT = int.method.RT
    local RX = int.method.RX
    local NP = int.method.basis.NP
    local NP = int.method.basis.NP

    local quad_b = int.method.grid_weights
    local C = cache(int,ST)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local brx = int.method.spatial_quadrature.weights
    local brt = int.method.time_quadrature.weights
    local mλ₀_x = int.method.mλ₀_x
    local mμ_t = int.method.mμ_t

    current_idx = 1
    for d in 1:D 
        for p in 1:NP[d]
            z = zero(ST)
            for rt in 1:RT
                for rx in 1:RX
                    z +=  quad_b[rt,rx] * 
                        ( x_domain * timestep(int) * C.∂L∂U_quad_values[d,rt,rx] * C.∂u∂P_quad_values[d][rt, rx,p]
                        + x_domain                 * C.∂L∂V_quad_values[d,rt,rx] * C.∂v∂P_quad_values[d][rt, rx,p]
                        + x_domain * timestep(int) * C.∂L∂W_quad_values[d,rt,rx] * C.∂w∂P_quad_values[d][rt, rx,p])
                end
            end
            for rx in 1:RX
                z+= x_domain * brx[rx] * (C.λ₀_quad_values[d,rx] * C.∂u∂P_t₀_quad_values[d][rx,p]) #- C.λ₁_quad_values[d,rx] * C.∂u∂P_t₁_quad_values[d, rx]
            end
            for rt in 1:RT
                z+= timestep(int)* brt[rt] * (C.μ₀_quad_values[d,rt] * C.∂u∂P_x₀_quad_values[d][rt,p] - C.μ₁_quad_values[d,rt] * C.∂u∂P_x₁_quad_values[d][rt,p])
            end
            b[current_idx] = z 
            current_idx += 1
        end
    end

    @assert current_idx == NP + 1 "Wrong indexing in residual computation"

    for d in 1:D
        for i in 1:RX
            z = zero(ST)
            for rx in 1:RX
                z += x_domain * brx[rx] * mλ₀_x[i,rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
            end
            b[NP + (d - 1) * RX + i] = z
        end
    end

    for d in 1:D
        for i in 1:RT
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) *brt[rt] * mμ_t[i,rt] *(C.ux₀_quad_values[d,rt] - C.boundary_condition_x₀[d,rt])
            end
            b[NP + D * RX + (d - 1) * RT + i] = z
        end
    end

    for d in 1:D
        for i in 1:RT
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) *brt[rt] * mμ_t[i,rt] *(C.boundary_condition_x₁[d,rt] - C.ux₁_quad_values[d,rt])
            end
            b[NP + D * RX + D * RT + (d - 1) * RT + i] = z
        end
    end
end


function residual!(b::Vector{ST}, sol, int::PDEIntegrator{IT}) where {ST,T, MVT <: Lagrange,LT  <: Lagrange,BT,IT <: NN_PDE_Integrator{T, MVT, LT, BT}}
    local D = int.problem.D 
    local RT = int.method.RT
    local RX = int.method.RX
    local NP = int.method.basis.NP

    local quad_b = int.method.grid_weights
    local C = cache(int,ST)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local brx = int.method.spatial_quadrature.weights
    local brt = int.method.time_quadrature.weights

    current_idx = 1
    for d in 1:D 
        for p in 1:NP
            z = zero(ST)
            for rt in 1:RT
                for rx in 1:RX
                    z +=  quad_b[rt,rx] * 
                        ( x_domain * timestep(int) * C.∂L∂U_quad_values[d,rt,rx] * C.∂u∂P_quad_values[d][rt, rx,p]
                        + x_domain                 * C.∂L∂V_quad_values[d,rt,rx] * C.∂v∂P_quad_values[d][rt, rx,p]
                        + x_domain * timestep(int) * C.∂L∂W_quad_values[d,rt,rx] * C.∂w∂P_quad_values[d][rt, rx,p])
                end
            end
            for rx in 1:RX
                z+= x_domain * brx[rx] * (C.λ₀_quad_values[d,rx] * C.∂u∂P_t₀_quad_values[d][rx,p]) #- C.λ₁_quad_values[d,rx] * C.∂u∂P_t₁_quad_values[d, rx]
            end
            for rt in 1:RT
                z+= timestep(int)* brt[rt] * (C.μ₀_quad_values[d,rt] * C.∂u∂P_x₀_quad_values[d][rt,p] - C.μ₁_quad_values[d,rt] * C.∂u∂P_x₁_quad_values[d][rt,p])
            end
            b[current_idx] = -z # TODO: check the sign
            current_idx += 1
        end
    end

    @assert current_idx == NP + 1 "Wrong indexing in residual computation"

    for d in 1:D
        for rx in 1:RX
            b[NP + (d - 1) * RX + rx] = x_domain * brx[rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
        end
    end

    for d in 1:D
        for rt in 1:RT
            b[NP+ D * RX+(d-1)*RT+rt]= timestep(int) *brt[rt]*(C.ux₀_quad_values[d,rt] - C.boundary_condition_x₀[d,rt])
        end
    end

    for d in 1:D
        for rt in 1:RT
            b[NP+ D * RX+ D * RT + (d-1)*RT+rt]= timestep(int) *brt[rt]*(C.ux₁_quad_values[d,rt] - C.boundary_condition_x₁[d,rt])
        end
    end

end



function update!(sol_struct, int::PDEIntegrator{<:NN_PDE_Integrator})
    local D = int.problem.D
    local u = [int.method.basis.u]
    local v = [int.method.basis.v]
    local w = [int.method.basis.w]
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local C = cache(int)
    local NP = int.method.basis.NP
    local x = nlsolution(int)
    local optim_mode = int.method.basis.optim_mode
    last_layer = keys(u[1].params)[end]

    x_nodes = collect(xspan[1]:xstep:xspan[2])
    if optim_mode == :Fully
        tem_params = NeuralNetworkParameters(reconstruct_params(x[1:NP], u[1].params))
        for (name, layer) in zip(keys(tem_params), values(tem_params))
            if hasfield(typeof(layer), :b)
                C.sol_params[name].W[:] = layer.W[:] 
                C.sol_params[name].b[:] = layer.b[:]
            else
                # For layers without bias (e.g., output), just regenerate W
                C.sol_params[name].W[:] = layer.W[:]
            end
        end
    elseif optim_mode == :Partially
        for (name, layer) in zip(keys(u[1].params), values(u[1].params))
            if hasfield(typeof(layer), :b)
                C.sol_params[name].W[:] = layer.W[:] 
                C.sol_params[name].b[:] = layer.b[:]
            else
                # For layers without bias (e.g., output), just regenerate W
                C.sol_params[name].W[:] = layer.W[:]
            end
        end
        C.sol_params[last_layer].W[:] = x[1:NP]
    end
    # println("In update! function, step = ", sol_struct.current_step)
    # println("In update! function, time = ", sol_struct.t)

    for d in 1:D
        for i in eachindex(x_nodes)
            sol_struct.sol.u[sol_struct.current_step][i] = u[d]([sol_struct.t, x_nodes[i]],C.sol_params)[1]
            sol_struct.sol.v[sol_struct.current_step][i] = v[d]([sol_struct.t, x_nodes[i]],C.sol_params)[1]
            sol_struct.sol.w[sol_struct.current_step][i] = w[d]([sol_struct.t, x_nodes[i]],C.sol_params)[1]
        end
    end

    # copy internal variables from cache to solution
    sol_struct.internal.x[sol_struct.current_step] .= cache(int).x 

    sol_struct.t += int.problem.tstep
    # println("In the end of update! function, time = ", sol_struct.t)
end

