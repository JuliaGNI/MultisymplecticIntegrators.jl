struct TrialNN_PDE_int{BT<:AbstractPDEBasis} <: PDEMethod
    basis::BT
    time_quadrature
    RT::Int # Number of quadrature points in time

    spatial_quadrature
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions

    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights

    N_in::Int # Inside the Domain

    function TrialNN_PDE_int(trial_NN; RT::Int=6, RX::Int=8, N_in::Int=600)
        if RT == 128
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

        dimensions = [RT, RX]
        grid_matrix, grid_weights = construct_quadrature_grid(dimensions)

        # number of collocation points during parameter initial initial_guess_method


        new{typeof(trial_NN)}(trial_NN,
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights, N_in)
    end
end


default_solver(::TrialNN_PDE_int) = Newton()

struct TrialNN_PDE_intCache{ST,RT,RX,D,NP} <: PDEIntegratorCache{ST,D}
    """
    RT = number of quadrature points in time
    RX = number of quadrature points in space
    D = dimension of output of u,v,w, i.e. scaler value function (D = 1) or vector function
    DX = dimension of input of u,v,w, i.e. number of spatial dimensions
    NP = number of parameters in the expression
    """
    x::Vector{ST}

    nn_quad_values::Array{ST}
    nnt_quad_values::Array{ST}
    nnx_quad_values::Array{ST}

    nn_t₀_quad_values::Array{ST}
    nn_x₀_quad_values::Array{ST}
    nn_x₁_quad_values::Array{ST}
    nn_t₀x₀::Matrix{ST}
    nn_t₀x₁::Matrix{ST}

    nnt_t₀_quad_values::Array{ST}
    nnt_x₀_quad_values::Array{ST}
    nnt_x₁_quad_values::Array{ST}
    nnt_t₀x₀::Matrix{ST}
    nnt_t₀x₁::Matrix{ST}

    nnx_t₀_quad_values::Array{ST}
    nnx_x₀_quad_values::Array{ST}
    nnx_x₁_quad_values::Array{ST}
    nnx_t₀x₀::Matrix{ST}
    nnx_t₀x₁::Matrix{ST}

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

    ∂u∂θ_quad_values::Array{ST}
    ∂v∂θ_quad_values::Array{ST}
    ∂w∂θ_quad_values::Array{ST}

    ut₀_quad_values::Matrix{ST} # bottom boundary, i.e. t = 0
    ux₀_quad_values::Matrix{ST} # left boundary, i.e. x = 0
    ux₁_quad_values::Matrix{ST} # right boundary, i.e. x = L

    basis_nn_ps
    
    ics_t₀_quad_values::Matrix{ST}
    bcs_x₀_quad_values::Matrix{ST}
    bcs_x₁_quad_values::Matrix{ST}
    bcs_t₀x₀_quad_values::Vector{ST}
    bcs_t₀x₁_quad_values::Vector{ST}

    icst_t₀_quad_values::Matrix{ST}
    bcst_x₀_quad_values::Matrix{ST}
    bcst_x₁_quad_values::Matrix{ST}
    bcst_t₀x₀_quad_values::Vector{ST}
    bcst_t₀x₁_quad_values::Vector{ST}

    icsx_t₀_quad_values::Matrix{ST}
    bcsx_x₀_quad_values::Matrix{ST}
    bcsx_x₁_quad_values::Matrix{ST}
    bcsx_t₀x₀_quad_values::Vector{ST}
    bcsx_t₀x₁_quad_values::Vector{ST}

    function TrialNN_PDE_intCache{ST,RT,RX,D,NP}() where {ST,RT,RX,D,NP}
        x = zeros(ST, NP) # in ELM, x is just the output layer parameters

        nn_quad_values = zeros(ST, D, RT, RX,NP)
        nnt_quad_values = zeros(ST, D, RT, RX, NP)
        nnx_quad_values = zeros(ST, D, RT, RX, NP)

        nn_t₀_quad_values = zeros(ST, D, RX, NP)
        nn_x₀_quad_values = zeros(ST, D, RT, NP)
        nn_x₁_quad_values = zeros(ST, D, RT, NP)
        nn_t₀x₀ = zeros(ST, D, NP)
        nn_t₀x₁ = zeros(ST, D, NP)

        nnt_t₀_quad_values = zeros(ST, D, RX, NP)
        nnt_x₀_quad_values = zeros(ST, D, RT, NP)
        nnt_x₁_quad_values = zeros(ST, D, RT, NP)
        nnt_t₀x₀ = zeros(ST, D, NP)
        nnt_t₀x₁ = zeros(ST, D, NP)

        nnx_t₀_quad_values = zeros(ST, D, RX, NP)
        nnx_x₀_quad_values = zeros(ST, D, RT, NP)
        nnx_x₁_quad_values = zeros(ST, D, RT, NP)
        nnx_t₀x₀ = zeros(ST, D, NP)
        nnx_t₀x₁ = zeros(ST, D, NP)

        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        ∂u∂θ_quad_values = zeros(ST, D, RT, RX, NP)
        ∂v∂θ_quad_values = zeros(ST, D, RT, RX, NP)
        ∂w∂θ_quad_values = zeros(ST, D, RT, RX, NP)

        ut₀_quad_values = zeros(ST, D, RX) # bottom boundary, i.e. t = 0
        ux₀_quad_values = zeros(ST, D, RT) # left boundary, i.e. x = 0
        ux₁_quad_values = zeros(ST, D, RT) # right boundary, i.e. x = L

        basis_nn_ps = (L1=(W=zeros(ST, 100, 2), b=zeros(ST, 100)), 
        L2=(W=zeros(ST, 100, 100),b=zeros(ST, 100)), 
        L3=(W=zeros(ST, 100, 100),b=zeros(ST, 100)),
        L4=(W=zeros(ST, NP, 100),b=zeros(ST, NP)),)

        ics_t₀_quad_values = zeros(ST, D, RX)
        bcs_x₀_quad_values = zeros(ST, D, RT)
        bcs_x₁_quad_values = zeros(ST, D, RT)
        bcs_t₀x₀_quad_values = zeros(ST, D)
        bcs_t₀x₁_quad_values = zeros(ST, D)

        icst_t₀_quad_values = zeros(ST, D, RX)
        bcst_x₀_quad_values = zeros(ST, D, RT)
        bcst_x₁_quad_values = zeros(ST, D, RT)
        bcst_t₀x₀_quad_values = zeros(ST, D)
        bcst_t₀x₁_quad_values = zeros(ST, D)

        icsx_t₀_quad_values = zeros(ST, D, RX)
        bcsx_x₀_quad_values = zeros(ST, D, RT)
        bcsx_x₁_quad_values = zeros(ST, D, RT)
        bcsx_t₀x₀_quad_values = zeros(ST, D)
        bcsx_t₀x₁_quad_values = zeros(ST, D)

        new(x,
            nn_quad_values,nnt_quad_values,nnx_quad_values,
            nn_t₀_quad_values,nn_x₀_quad_values,nn_x₁_quad_values,
            nn_t₀x₀,nn_t₀x₁,
            nnt_t₀_quad_values,nnt_x₀_quad_values,nnt_x₁_quad_values,
            nnt_t₀x₀,nnt_t₀x₁,
            nnx_t₀_quad_values,nnx_x₀_quad_values,nnx_x₁_quad_values,
            nnx_t₀x₀,nnx_t₀x₁,
            u_quad_values,v_quad_values,w_quad_values,
            ∂L∂U_quad_values,∂L∂V_quad_values,∂L∂W_quad_values,
            ∂u∂θ_quad_values,∂v∂θ_quad_values,∂w∂θ_quad_values,
            ut₀_quad_values,ux₀_quad_values,ux₁_quad_values,
            basis_nn_ps,
            ics_t₀_quad_values,bcs_x₀_quad_values,bcs_x₁_quad_values,
            bcs_t₀x₀_quad_values,bcs_t₀x₁_quad_values,
            icst_t₀_quad_values,bcst_x₀_quad_values,bcst_x₁_quad_values,
            bcst_t₀x₀_quad_values,bcst_t₀x₁_quad_values,
            icsx_t₀_quad_values,bcsx_x₀_quad_values,bcsx_x₁_quad_values,
            bcsx_t₀x₀_quad_values,bcsx_t₀x₁_quad_values
            )
    end
end

nlsolution(cache::TrialNN_PDE_intCache) = cache.x

function Cache{ST}(problem::PDEProblem, int::TrialNN_PDE_int; kwargs...) where {ST}
    TrialNN_PDE_intCache{ST,int.RT,int.RX,problem.D,int.basis.NP}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::PDEProblem, int::TrialNN_PDE_int) = TrialNN_PDE_intCache{ST,int.RT,int.RX,problem.D,int.basis.NP}

@inline function Base.getindex(c::TrialNN_PDE_intCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end


function T1NN(t, x, tn, params,int)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local h = timestep(int)
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local PNN = int.method.basis.basis_network

    return (b - x) / x_domain * PNN([t,a],params)[1] +
       (x - a) / x_domain * PNN([t,b],params)[1] +
        (h - t) / h * PNN([0.0,x],params)[1]
end

function T2NN(t, x, tn, params,int)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local h = timestep(int)
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local PNN = int.method.basis.basis_network

    return (b - x) / x_domain  *(h - t) / h * PNN([0.0,a],params)[1] +
        (x - a) / x_domain * (h - t) / h * PNN([0.0,b],params)[1]
end

function C1(t, x, tn, params,int)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local h = timestep(int)
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local PNN = int.method.basis.basis_network
    if current_step == 1
        return (b - x) / x_domain * bc_fun(t, xspan).bc₀.u +
            (x - a) / x_domain * bc_fun(t, xspan).bc₁.u +
            (h - t) / h * ic_fun(x).u
    else
        return (b - x) / x_domain * bc_fun(t, xspan).bc₀.u +
            (x - a) / x_domain * bc_fun(t, xspan).bc₁.u +
            (h - t) / h * PNN([1.0,x],previous_params)[1]
    end
end



function C2(t, x, tn, params)
    return psi_L(x) * phi_B(t) * bc_fun(tn, xspan).bc₀.u +
        psi_R(x) * phi_B(t) * bc_fun(tn, xspan).bc₁.u
end

u_trial(t,x,tn,params) = PNN([t,x],params)[1] - T1NN(t,x,tn,params) + T2NN(t,x,tn,params) + C1(t,x,tn,params) - C2(t,x,tn,params)

function prior_initial_guess!(C, sol, int::PDEIntegrator{<:TrialNN_PDE_int})
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local xspan = int.problem.xspan

    local h = timestep(int)
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local PNN = int.method.basis.sol_network
    local BNN = int.method.basis.basis_network
    local ic_fun = int.problem.ics_function
    local bc_fun = int.problem.bcs_function
    local previous_params = sol.internal.previous_params
    local current_step = sol.current_step
    local exact_u = int.problem.exact_u
    psi_L(x) = (b - x) / x_domain       # lefts
    psi_R(x) = (x - a) / x_domain       # right
    phi_B(t) = (h - t) / h   # bottom

    function mse_loss(params, tx_in, u_trial)
        loss = 0.0
        for i in 1:size(tx_in, 2)
            t, x = tx_in[:, i]
            pred = u_trial(t, x, 0.0, params)
            label = exact_u(t, x)
            loss += (pred - label)^2
        end
        return loss / size(tx_in, 2)
    end

    #use exact_sol for initialization temporarily for proof of concept
    # 5000 random points inside the domain
    tx_in = rand(Random.seed!(1),2,5000)
    tx_in[2,:] .= xspan[1] .+ (xspan[2] - xspan[1]) * tx_in[2,:]

    epochs = 10000
    opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.AdamOptimizerWithDecay(epochs), PNN.params)
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

            grads = Zygote.gradient(d -> mse_loss(d, batch_tx,u_trial), PNN.params)[1]
            GeometricMachineLearning.optimization_step!(opt, λ, PNN.params, grads)
            batch_loss = mse_loss(PNN.params, batch_tx,u_trial)
            epoch_loss += batch_loss * size(batch_tx, 2)
        end
        epoch_loss /= num_samples
        push!(loss_history, epoch_loss)
        println("Epoch $epoch, MSE Loss: $epoch_loss")
        if epoch_loss < 1e-5
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

    local RT = int.method.RT
    local RX = int.method.RX
    local D = int.problem.D
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes

    BNNt(t,x) = Zygote.gradient(tt -> BNN([tt,x], BNN.params)[1], t)[1]
    BNNx(t,x) = Zygote.gradient(xx -> BNN([t,xx], BNN.params)[1], x)[1]
    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                C.nn_quad_values[d,rt,rx,:] = BNN([t_quad_nodes[rt], x_quad_nodes[rx]], BNN.params)[1]
                #TODO:check if this is correct with cache parameters
                C.nnt_quad_values[d,rt,rx,:] = BNNt(t_quad_nodes[rt], x_quad_nodes[rx])
                C.nnx_quad_values[d,rt,rx,:] = BNNx(t_quad_nodes[rt], x_quad_nodes[rx])
            end
        end
        for rx in 1:RX
            C.nn_t₀_quad_values[d,rx,:] = BNN([0.0, x_quad_nodes[rx]], BNN.params)[1]
            C.nnt_t₀_quad_values[d,rx,:] = BNNt(0.0, x_quad_nodes[rx])
            C.nnx_t₀_quad_values[d,rx,:] = BNNx(0.0, x_quad_nodes[rx])
        end

        for rt in 1:RT
            C.nn_x₀_quad_values[d,rt,:] = BNN([t_quad_nodes[rt], a], BNN.params)[1]
            C.nn_x₁_quad_values[d,rt,:] = BNN([t_quad_nodes[rt], b], BNN.params)[1]
            C.nnt_x₀_quad_values[d,rt,:] = BNNt(t_quad_nodes[rt], a)
            C.nnt_x₁_quad_values[d,rt,:] = BNNt(t_quad_nodes[rt], b)
            C.nnx_x₀_quad_values[d,rt,:] = BNNx(t_quad_nodes[rt], a)
            C.nnx_x₁_quad_values[d,rt,:] = BNNx(t_quad_nodes[rt], b)
        end

        C.nn_t₀x₀[d,:] = BNN([0.0, a], BNN.params)
        C.nn_t₀x₁[d,:] = BNN([0.0, b], BNN.params)
        C.nnt_t₀x₀[d,:] = BNNt(0.0, a)
        C.nnt_t₀x₁[d,:] = BNNt(0.0, b)
        C.nnx_t₀x₀[d,:] = BNNx(0.0, a)
        C.nnx_t₀x₁[d,:] = BNNx(0.0, b)
    end

end

function initialize_bcs_ics!(sol,int::PDEIntegrator{<:TrialNN_PDE_int}) 
    local C = cache(int)
    local current_step = sol.current_step
    local D = int.problem.D
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local ic_fun = int.problem.ics_function
    local bc_fun = int.problem.bcs_function
    local xspan = int.problem.xspan
    if current_step == 1
        for d in 1:D
            C.ics_t₀_quad_values[d,:] = ic_fun(x_quad_nodes).u
            C.icst_t₀_quad_values[d,:] = ic_fun(x_quad_nodes).v
            C.icsx_t₀_quad_values[d,:] = ic_fun(x_quad_nodes).w
        end
    else
        for d in 1:D
            C.ics_t₀_quad_values[d,:]  = sol.internal.u_quad_end_values[d,:]
            C.icst_t₀_quad_values[d,:] = sol.internal.v_quad_end_values[d,:]
            C.icsx_t₀_quad_values[d,:] = sol.internal.w_quad_end_values[d,:]
        end
    end

    for d in 1:D
        C.bcs_x₀_quad_values[d,:] = bc_fun(t_quad_nodes, xspan).bc₀.u
        C.bcs_x₁_quad_values[d,:] = bc_fun(t_quad_nodes, xspan).bc₁.u
        C.bcs_t₀x₀[d,:] = bc_fun(0.0, xspan).bc₀.u
        C.bcs_t₀x₁[d,:] = bc_fun(0.0, xspan).bc₁.u
    
        C.bcst_x₀_quad_values[d,:] = bc_fun(t_quad_nodes, xspan).bc₀.v
        C.bcst_x₁_quad_values[d,:] = bc_fun(t_quad_nodes, xspan).bc₁.v
        C.bcst_t₀x₀[d,:] = bc_fun(0.0, xspan).bc₀.v
        C.bcst_t₀x₁[d,:] = bc_fun(0.0, xspan).bc₁.v

        C.bcsx_x₀_quad_values[d,:] = bc_fun(t_quad_nodes, xspan).bc₀.w
        C.bcsx_x₁_quad_values[d,:] = bc_fun(t_quad_nodes, xspan).bc₁.w
        C.bcsx_t₀x₀[d,:] = bc_fun(0.0, xspan).bc₀.w
        C.bcsx_t₀x₁[d,:] = bc_fun(0.0, xspan).bc₁.w
    end
end

#  Trial solution function construction
# psi_L(x) = (b - x) / x_domain       # left 
# psi_R(x) = (x - a) / x_domain       # right
# phi_B(t) = (h - t) / h   # bottom

# function T1NN(t, x, tn, dofs, int)
#     local fixed_params = cache(int).basis_nn_ps
#     local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
#     local h = timestep(int)
#     local a,b = int.problem.xspan[1],int.problem.xspan[2]
#     local PNN = int.method.basis.basis_network

#     return (b - x)  * dofs' * PNN([t,a], fixed_params) / x_domain +
#         (x - a)  * dofs' * PNN([t,b],fixed_params) / x_domain +
#         (h - t)  * dofs' * PNN([0.0,x],fixed_params) / h
# end

# function T2NN(t, x, tn, dofs, int)
#     local fixed_params = cache(int).basis_nn_ps
#     local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
#     local h = timestep(int)
#     local a,b = int.problem.xspan[1],int.problem.xspan[2]
#     local PNN = int.method.basis.basis_network

#     return (b - x)   * (h - t)  * dofs' * PNN([0.0,a],fixed_params) / x_domain / h +
#         (x - a)  * (h - t)  * dofs' * PNN([0.0,b],fixed_params)/ x_domain / h
# end

# function C1(t, x, tn, dofs,int,sol)
#     local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
#     local h = timestep(int)
#     local a,b = int.problem.xspan[1],int.problem.xspan[2]
#     local previous_params = sol.internal.previous_params
#     local PNN = int.method.basis.basis_network
#     local bc_fun = int.problem.bcs_function
#     local ic_fun = int.problem.ics_function
#     local current_step = sol.current_step
#     local xspan = int.problem.xspan

#     if current_step == 1
#         return (b - x)  * bc_fun(t, xspan).bc₀.u / x_domain+
#             (x - a)  * bc_fun(t, xspan).bc₁.u / x_domain +
#             (h - t)  * ic_fun(x).u / h
#     else
#         return (b - x)  * bc_fun(t, xspan).bc₀.u / x_domain +
#             (x - a)  * bc_fun(t, xspan).bc₁.u / x_domain +
#             (h - t) * sol.internal.x[current_step-1]' * PNN([1.0,x],previous_params) / h
#     end
# end

# function C2(t, x, tn, dofs,int)
#     local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
#     local a,b = int.problem.xspan[1],int.problem.xspan[2]
#     local h = timestep(int)
#     local bc_fun = int.problem.bcs_function
#     local xspan = int.problem.xspan
#     return (b - x)  * (h - t) * bc_fun(tn, xspan).bc₀.u / x_domain / h +
#         (x - a)  * (h - t) * bc_fun(tn, xspan).bc₁.u / x_domain / h
# end


# # u_trial(t,x,tn,dofs,int) = sum(dofs .*PNN([t,x],fixed_params)) - T1NN(t,x,tn,dofs,int) + T2NN(t,x,tn,dofs,int) + C1(t,x,tn,dofs,int) - C2(t,x,tn,dofs,int)

# function u_trial(t,x,tn,dofs,int,sol)
#     local fixed_params = cache(int).basis_nn_ps
#     local PNN = int.method.basis.basis_network
#     return dofs' * PNN([t,x],fixed_params) - T1NN(t,x,tn,dofs,int) + T2NN(t,x,tn,dofs,int) + C1(t,x,tn,dofs,int,sol) - C2(t,x,tn,dofs,int)
# end

# v_trial(t,x,tn,dofs,int,sol) = Zygote.gradient(tt -> u_trial(tt,x,tn,dofs,int,sol),t)[1]
# w_trial(t,x,tn,dofs,int,sol) = Zygote.gradient(xx -> u_trial(t,xx,tn,dofs,int,sol),x)[1]

# ∂u∂θ_func(t,x,tn,dofs,int,sol) = Zygote.gradient(θ -> u_trial(t,x,tn,θ,int,sol), dofs)[1]
# ∂v∂θ_func(t,x,tn,dofs,int,sol) = Zygote.gradient(θ -> v_trial(t,x,tn,θ,int,sol), dofs)[1]
# ∂w∂θ_func(t,x,tn,dofs,int,sol) = Zygote.gradient(θ -> w_trial(t,x,tn,θ,int,sol), dofs)[1]


function components!(x::AbstractVector{ST}, sol, int::PDEIntegrator{<:TrialNN_PDE_int}) where {ST}
    local problem_params = int.problem.lagrangian_system.params
    local grid_matrix = int.method.grid_matrix
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes

    local ∂L∂U = int.problem.lagrangian_system.functions.∂L∂U
    local ∂L∂V = int.problem.lagrangian_system.functions.∂L∂V
    local ∂L∂W = int.problem.lagrangian_system.functions.∂L∂W
    local D = int.problem.D 
    local RT = int.method.RT
    local RX = int.method.RX
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local C = cache(int,ST)

    local h = timestep(int)
    local current_step = sol.current_step
    local tn = (current_step - 1) * h
 

    # function nlls!(du, u, int::PDEIntegrator{<:TrialNN_PDE_int})
    #     local xspan = int.problem.xspan
    #     local c = int.problem.params.c
    #     local tn = (sol.current_step - 1) * timestep(int)
    #     local N_in = int.method.N_in

    #     tx_in = rand(Random.seed!(1),2,N_in)
    #     tx_in[2,:] .= xspan[1] .+ (xspan[2] - xspan[1]) * tx_in[2,:]

    #     for i in 1:N_in
    #         du[i] = v_trial(tx_in[1,i], tx_in[2,i], tn, u) + c * w_trial(tx_in[1,i], tx_in[2,i], tn, u)
    #     end
    # end

    # if C.done_initial_guess[1] == 0
    #     u0 = zeros(1,NP)
    #     prob = NonlinearLeastSquaresProblem(
    #     NonlinearFunction(nlls!, resid_prototype = zeros(N_in)), u0, int)
    #     println("Starting initial guess computation ...")
    #     t1 = time()
    #     u_sol = solve(prob,maxtime = 60,abstol = 1e-12, reltol = 1e-12).u
    #     (x == cache(int).x && eltype(x) == eltype(cache(int).x)) ? x[:] = u_sol : nothing
    #     println("Time for initial guess: ", time() - t1)
    #     print("initial guess parameters: ", x, "\n")
    #     C.done_initial_guess[1] = 1
    # end

    # if C.done_initial_guess[1] == 0
    #     prior_initial_guess!(C, sol, int) 
    #     C.done_initial_guess[1] = 1
    # end

    # Reconstruct C1 part for u,v,W
    t1 = time()

    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                C.∂u∂θ_quad_values[d,rt,rx,:] = C.nn_quad_values[d,rt,rx,:] 
                - (x_quad_nodes[RX-rx] * C.nn_x₀_quad_values[d,rt,:] + x_quad_nodes[rx]*C.nn_x₁_quad_values[d,rt,:] + (t_quad_nodes[RT-rt] * C.nn_t₀_quad_values[d,rx,:]))
                + (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.nn_t₀x₀[d,:] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.nn_t₀x₁[d,:])
                
                C.∂v∂θ_quad_values[d,rt,rx,:] = C.nnt_quad_values[d,rt,rx,:]
                - (x_quad_nodes[RX-rx] * C.nnt_x₀_quad_values[d,rt,:] + x_quad_nodes[rx]*C.nnt_x₁_quad_values[d,rt,:] + (t_quad_nodes[RT-rt] * C.nnt_t₀_quad_values[d,rx,:]))
                + (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.nnt_t₀x₀[d,:] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.nnt_t₀x₁[d,:])

                C.∂w∂θ_quad_values[d,rt,rx,:] = C.nnw_quad_values[d,rt,rx,:]
                - (x_quad_nodes[RX-rx] * C.nnw_x₀_quad_values[d,rt,:] + x_quad_nodes[rx]*C.nnw_x₁_quad_values[d,rt,:] + (t_quad_nodes[RT-rt] * C.nnw_t₀_quad_values[d,rx,:]))
                + (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.nnw_t₀x₀[d,:] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.nnw_t₀x₁[d,:])
            end
        end
    end
    t2 = time()
    println("Time for computing ∂u/∂θ, ∂v/∂θ, ∂w/∂θ at quad points: ", t2 - t1)
    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                C.u_quad_values[d, rt, rx] = C.∂u∂θ_quad_values[d,rt,rx,:] * x 
                + (x_quad_nodes[RX-rx] * C.bcs_x₀_quad_values[d,rt] + x_quad_nodes[rx] * C.bcs_x₁_quad_values[d,rt] + t_quad_nodes[RT-rt] * C.ics_t₀_quad_values[d,rx])
                - (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.bcs_t₀x₀[d] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.bcs_t₀x₁[d])

                C.v_quad_values[d, rt, rx] = C.∂v∂θ_quad_values[d,rt,rx,:]' * x
                + (x_quad_nodes[RX-rx] * C.bcst_x₀_quad_values[d,rt] + x_quad_nodes[rx] * C.bcst_x₁_quad_values[d,rt] + t_quad_nodes[RT-rt] * C.icst_t₀_quad_values[d,rx])
                - (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.bcst_t₀x₀[d] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.bcst_t₀x₁[d])

                C.w_quad_values[d, rt, rx] = C.∂w∂θ_quad_values[d,rt,rx,:]' * x
                + (x_quad_nodes[RX-rx] * C.bcsx_x₀_quad_values[d,rt] + x_quad_nodes[rx] * C.bcsx_x₁_quad_values[d,rt] + t_quad_nodes[RT-rt] * C.icsx_t₀_quad_values[d,rx])
                - (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.bcsx_t₀x₀[d] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.bcsx_t₀x₁[d])              
            end
        end
    end
    t3 = time()
    println("Time for computing u,v,w at quad points: ", t3 - t2)

    for d in 1:D
        for rx in 1:RX
            C.ut₀_quad_values[d,rx] = C.nn_t₀_quad_values[d,rx,:]' * x 
                - (x_quad_nodes[RX-rx] * C.nn_x₀_quad_values[d,rt,:]'*x + x_quad_nodes[rx]*C.nn_x₁_quad_values[d,rt,:]'*x + (t_quad_nodes[RT-rt] * C.nn_t₀_quad_values[d,rx,:]'*x))
                + (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.nn_t₀x₀[d,:]'*x + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.nn_t₀x₁[d,:]'*x)
                + (x_quad_nodes[RX-rx] * C.bcs_x₀_quad_values[d,rt] + x_quad_nodes[rx] * C.bcs_x₁_quad_values[d,rt] + t_quad_nodes[RT-rt] * C.ics_t₀_quad_values[d,rx])
                - (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.bcs_t₀x₀[d] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.bcs_t₀x₁[d])
        end

        for rt in 1:RT
            C.ux₀_quad_values[d,rx] =  C.nn_x₀_quad_values[d,rt,:]' * x 
                - (x_quad_nodes[RX-rx] * C.nn_x₀_quad_values[d,rt,:]'*x + x_quad_nodes[rx]*C.nn_x₁_quad_values[d,rt,:]'*x + (t_quad_nodes[RT-rt] * C.nn_t₀_quad_values[d,rx,:]'*x))
                + (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.nn_t₀x₀[d,:]'*x + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.nn_t₀x₁[d,:]'*x)
                + (x_quad_nodes[RX-rx] * C.bcs_x₀_quad_values[d,rt] + x_quad_nodes[rx] * C.bcs_x₁_quad_values[d,rt] + t_quad_nodes[RT-rt] * C.ics_t₀_quad_values[d,rx])
                - (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.bcs_t₀x₀[d] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.bcs_t₀x₁[d])

            C.ux₁_quad_values[d,i] =  C.ux₁_quad_values[d,i] =  C.nn_x₀_quad_values[d,rt,:]' * x 
                - (x_quad_nodes[RX-rx] * C.nn_x₀_quad_values[d,rt,:]'*x + x_quad_nodes[rx]*C.nn_x₁_quad_values[d,rt,:]'*x + (t_quad_nodes[RT-rt] * C.nn_t₀_quad_values[d,rx,:]'*x))
                + (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.nn_t₀x₀[d,:]'*x + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.nn_t₀x₁[d,:]'*x)
                + (x_quad_nodes[RX-rx] * C.bcs_x₀_quad_values[d,rt] + x_quad_nodes[rx] * C.bcs_x₁_quad_values[d,rt] + t_quad_nodes[RT-rt] * C.ics_t₀_quad_values[d,rx])
                - (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.bcs_t₀x₀[d] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.bcs_t₀x₁[d])
        end
    end

    t4 = time()
    println("Time for computing u,v,w at boundary quad points: ", t4 - t3)

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], problem_params)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], problem_params)
                C.∂L∂W_quad_values[d, i, j] = ∂L∂W[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], problem_params)
            end
        end 
    end
    t5 = time()
    println("Time for computing ∂L/∂U, ∂L/∂V, ∂L/∂W at quad points: ", t5 - t4)
    println("Total time for components! function: ", t5 - t1)
end

post_initial_guess!(C, sol, int::PDEIntegrator{<:TrialNN_PDE_int}) = nothing

function residual!(b::Vector{ST}, sol, int::PDEIntegrator{<:TrialNN_PDE_int}) where {ST}
    local D = int.problem.D 
    local RT = int.method.RT
    local RX = int.method.RX
    local NP = int.method.basis.NP

    local quad_b = int.method.grid_weights
    local C = cache(int,ST)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]

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

end

function update!(sol_struct, int::PDEIntegrator{<:TrialNN_PDE_int})
    local D = int.problem.D
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local x = nlsolution(int)
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local v_basis_func = int.method.basis.v
    local w_basis_func = int.method.basis.w
    local u_basis_func = int.method.basis.u
    local nn_params = int.method.basis.u.params
    local RX = int.method.RX
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    x_nodes = collect(xspan[1]:xstep:xspan[2])
    for d in 1:D
        for i in eachindex(x_nodes)
            # sol_struct.sol.u[sol_struct.current_step][i] = 
        end
    end
    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                C.u_quad_values[d, rt, rx] = C.∂u∂θ_quad_values[d,rt,rx,:] * x 
                + (x_quad_nodes[RX-rx] * C.bcs_x₀_quad_values[d,rt] + x_quad_nodes[rx] * C.bcs_x₁_quad_values[d,rt] + t_quad_nodes[RT-rt] * C.ics_t₀_quad_values[d,rx])
                - (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.bcs_t₀x₀[d] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.bcs_t₀x₁[d])

                C.v_quad_values[d, rt, rx] = C.∂v∂θ_quad_values[d,rt,rx,:]' * x
                + (x_quad_nodes[RX-rx] * C.bcst_x₀_quad_values[d,rt] + x_quad_nodes[rx] * C.bcst_x₁_quad_values[d,rt] + t_quad_nodes[RT-rt] * C.icst_t₀_quad_values[d,rx])
                - (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.bcst_t₀x₀[d] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.bcst_t₀x₁[d])

                C.w_quad_values[d, rt, rx] = C.∂w∂θ_quad_values[d,rt,rx,:]' * x
                + (x_quad_nodes[RX-rx] * C.bcsx_x₀_quad_values[d,rt] + x_quad_nodes[rx] * C.bcsx_x₁_quad_values[d,rt] + t_quad_nodes[RT-rt] * C.icsx_t₀_quad_values[d,rx])
                - (x_quad_nodes[RX-rx] * t_quad_nodes[RT-rt] * C.bcsx_t₀x₀[d] + x_quad_nodes[rx] * t_quad_nodes[RT-rt] * C.bcsx_t₀x₁[d])              
            end
        end
    end

    # copy internal variables from cache to solution
    sol_struct.t = (sol_struct.current_step+1) * timestep(int)
    # println("In the end of update! function, time = ", sol_struct.t)
end

function internal_variables(int::PDEIntegrator{<:TrialNN_PDE_int}, problem::PDEProblem)
    local PNN = int.method.basis.sol_network
    local BNN = int.method.basis.basis_network
    local C = cache(int)
    local x = nlsolution(int)
    local RX = int.method.RX
    local D = problem.D
    local h = timestep(int)

    for (name, layer) in zip(keys(PNN.params), values(PNN.params))
        if hasfield(typeof(layer), :b)
            layer.W[:] = C.basis_nn_ps[name].W[:] 
            layer.b[:] = C.basis_nn_ps[name].b[:]
        else
            # For layers without bias (e.g., output), just regenerate W
            layer.W[:] = x
        end
    end

    u_quad_end_values = zeros(D,RX)
    v_quad_end_values = zeros(D,RX)
    w_quad_end_values = zeros(D,RX)
    # for d in 1:D 
    #     for rx in 1:RX
    #         u_quad_end_values[d,rx] = u_trial(1.0, spatial_quad_nodes[rx], tn, x, int,sol)
    #         v_quad_end_values[d,rx] = v_trial(1.0, spatial_quad_nodes[rx], tn, x, int,sol)
    #         w_quad_end_values[d,rx] = w_trial(1.0, spatial_quad_nodes[rx], tn, x, int,sol)
    #     end 
    # end


    return (previous_params = PNN.params,
        u_quad_end_values = u_quad_end_values,
        v_quad_end_values = v_quad_end_values,
        w_quad_end_values = w_quad_end_values
        )
end

