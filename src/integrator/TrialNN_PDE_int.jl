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

    done_initial_guess::Vector{Int} # flag for initial guess computation
    function TrialNN_PDE_intCache{ST,RT,RX,D,NP}() where {ST,RT,RX,D,NP}
        x = zeros(ST, NP) # in ELM, x is just the output layer parameters

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

        basis_nn_ps = (L1=(W=zeros(ST, 100, 2), b=zeros(ST, 100)), L2=(W=zeros(ST, 100, 100),b=zeros(ST, 100)), 
        L3=(W=zeros(ST, 100, 100),b=zeros(ST, 100)),L4=(W=zeros(ST, NP, 100),b=zeros(ST, NP)),) #L4=(W=zeros(ST, NP, 100),b=zeros(ST, NP)),

        done_initial_guess = [0]
        new(x,
            u_quad_values,
            v_quad_values,
            w_quad_values,
            ∂L∂U_quad_values,
            ∂L∂V_quad_values,
            ∂L∂W_quad_values,
            ∂u∂θ_quad_values,
            ∂v∂θ_quad_values,
            ∂w∂θ_quad_values,
            ut₀_quad_values,
            ux₀_quad_values,
            ux₁_quad_values,
            basis_nn_ps,
            done_initial_guess)
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

function prior_initial_guess!(C, sol, int::PDEIntegrator{<:TrialNN_PDE_int})
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local xspan = int.problem.xspan

    local h = timestep(int)
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local PNN = int.method.basis.sol_network
    local ic_fun = int.problem.ics_function
    local bc_fun = int.problem.bcs_function
    local previous_params = sol.internal.previous_params
    local current_step = sol.current_step
    local exact_u = int.problem.exact_u
    psi_L(x) = (b - x) / x_domain       # lefts
    psi_R(x) = (x - a) / x_domain       # right
    phi_B(t) = (h - t) / h   # bottom

    
    function T1NN(t, x, tn, params)
        return psi_L(x) * PNN([t,a],params)[1] +
            psi_R(x) * PNN([t,b],params)[1] +
            phi_B(t) * PNN([0.0,x],params)[1]
    end

    function T2NN(t, x, tn, params)
        return psi_L(x) * phi_B(t) * PNN([0.0,a],params)[1] +
            psi_R(x) * phi_B(t) * PNN([0.0,b],params)[1]
    end

    function C1(t, x, tn, params)
        if current_step == 1
            return psi_L(x) * bc_fun(t, xspan).bc₀.u +
                psi_R(x) * bc_fun(t, xspan).bc₁.u +
                phi_B(t) * ic_fun(x).u 
        else
            return psi_L(x) * bc_fun(t, xspan).bc₀.u +
                psi_R(x) * bc_fun(t, xspan).bc₁.u +
                phi_B(t) * PNN([1.0,x],previous_params)[1]
        end
    end

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


    function C2(t, x, tn, params)
        return psi_L(x) * phi_B(t) * bc_fun(tn, xspan).bc₀.u +
            psi_R(x) * phi_B(t) * bc_fun(tn, xspan).bc₁.u
    end

    u_trial(t,x,tn,params) = PNN([t,x],params)[1] - T1NN(t,x,tn,params) + T2NN(t,x,tn,params) + C1(t,x,tn,params) - C2(t,x,tn,params)

    #use exact_sol for initialization temporarily for proof of concept
    # 5000 random points inside the domain
    tx_in = rand(Random.seed!(1),2,5000)
    tx_in[2,:] .= xspan[1] .+ (xspan[2] - xspan[1]) * tx_in[2,:]

    epochs = 3000
    opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.AdamOptimizerWithDecay(epochs), PNN.params)
    λ = GeometricMachineLearning.GlobalSection(PNN.params)
    loss_history = []

    batch_size = 100
    num_samples = size(tx_in, 2)
    num_batches = cld(num_samples, batch_size)

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
        else
            # For layers without bias (e.g., output), just regenerate W
            C.x[:] = layer.W[:]
        end
    end

end

initialize_bcs_ics!(sol,int::PDEIntegrator{<:TrialNN_PDE_int}) = nothing


function components!(x::AbstractVector{ST}, sol, int::PDEIntegrator{<:TrialNN_PDE_int}) where {ST}
    local PNN = int.method.basis.basis_network
    local problem_params = int.problem.lagrangian_system.params

    local NP = int.method.basis.NP
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
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local ic_fun = int.problem.ics_function
    local bc_fun = int.problem.bcs_function
    local current_step = sol.current_step
    local tn = (current_step - 1) * h
    local N_in = int.method.N_in
    local previous_params = sol.internal.previous_params
 
    #  Trial solution function construction
    psi_L(x) = (b - x) / x_domain       # left 
    psi_R(x) = (x - a) / x_domain       # right
    phi_B(t) = (h - t) / h   # bottom

    fixed_params = cache(int).basis_nn_ps
    function T1NN(t, x, tn, dofs)
        return psi_L(x) * sum(dofs .* PNN([t,a],fixed_params)) +
            psi_R(x) * sum(dofs .* PNN([t,b],fixed_params)) +
            phi_B(t) * sum(dofs .* PNN([0.0,x],fixed_params))
    end

    function T2NN(t, x, tn, dofs)
        return psi_L(x) * phi_B(t) * sum(dofs .* PNN([0.0,a],fixed_params)) +
            psi_R(x) * phi_B(t) * sum(dofs .* PNN([0.0,b],fixed_params))
    end

    function C1(t, x, tn, dofs)
        if current_step == 1
            return psi_L(x) * bc_fun(t, xspan).bc₀.u +
                psi_R(x) * bc_fun(t, xspan).bc₁.u +
                phi_B(t) * ic_fun(x).u 
        else
            return psi_L(x) * bc_fun(t, xspan).bc₀.u +
                psi_R(x) * bc_fun(t, xspan).bc₁.u +
                phi_B(t) * sum(sol.internal.x[current_step-1] .* PNN([1.0,x],previous_params))
        end
    end

    function C2(t, x, tn, dofs)
        return psi_L(x) * phi_B(t) * bc_fun(tn, xspan).bc₀.u +
            psi_R(x) * phi_B(t) * bc_fun(tn, xspan).bc₁.u
    end


    u_trial(t,x,tn,dofs) = sum(dofs .*PNN([t,x],fixed_params)) - T1NN(t,x,tn,dofs) + T2NN(t,x,tn,dofs) + C1(t,x,tn,dofs) - C2(t,x,tn,dofs)
    v_trial(t,x,tn,dofs) = Zygote.gradient(tt -> u_trial(tt,x,tn,dofs),t)[1]
    w_trial(t,x,tn,dofs) = Zygote.gradient(xx -> u_trial(t,xx,tn,dofs),x)[1]

    ∂u∂θ_func(t,x,tn,dofs) = Zygote.gradient(θ -> u_trial(t,x,tn,θ), dofs)[1]
    ∂v∂θ_func(t,x,tn,dofs) = Zygote.gradient(θ -> v_trial(t,x,tn,θ), dofs)[1]
    ∂w∂θ_func(t,x,tn,dofs) = Zygote.gradient(θ -> w_trial(t,x,tn,θ), dofs)[1]

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

    if C.done_initial_guess[1] == 0
        print("Start initial guess function in Component function! \n")
        prior_initial_guess!(C, sol, int) 
        C.done_initial_guess[1] = 1
    end

    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                C.u_quad_values[d, rt, rx] = u_trial(grid_matrix[rt, rx][1], xspan[1] + x_domain* grid_matrix[rt, rx][2], tn, x)
                C.v_quad_values[d, rt, rx] = v_trial(grid_matrix[rt, rx][1], xspan[1] + x_domain* grid_matrix[rt, rx][2], tn, x)
                C.w_quad_values[d, rt, rx] = w_trial(grid_matrix[rt, rx][1], xspan[1] + x_domain* grid_matrix[rt, rx][2], tn, x)
            end
        end
    end

    for d in 1:D
        for j in 1:RX
            C.ut₀_quad_values[d,j] = u_trial(0.0 ,xspan[1] + x_domain* x_quad_nodes[j], tn, x)
        end

        for i in 1:RT
            C.ux₀_quad_values[d,i] =  u_trial(t_quad_nodes[i], xspan[1], tn, x)
            C.ux₁_quad_values[d,i] =  u_trial(t_quad_nodes[i], xspan[2], tn, x)
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

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.∂u∂θ_quad_values[d,i,j,:] = ∂u∂θ_func(grid_matrix[i,j][1], xspan[1] + x_domain* grid_matrix[i,j][2], tn, x)
                C.∂v∂θ_quad_values[d,i,j,:] = ∂v∂θ_func(grid_matrix[i,j][1], xspan[1] + x_domain* grid_matrix[i,j][2], tn, x)
                C.∂w∂θ_quad_values[d,i,j,:] = ∂w∂θ_func(grid_matrix[i,j][1], xspan[1] + x_domain* grid_matrix[i,j][2], tn, x)
            end
        end
    end

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
            sol_struct.sol.u[sol_struct.current_step][i] = sum(u_basis_func([sol_struct.t, x_nodes[i]],nn_params) .* x)
            sol_struct.sol.v[sol_struct.current_step][i] = sum(v_basis_func([sol_struct.t, x_nodes[i]],nn_params) .* x)
            sol_struct.sol.w[sol_struct.current_step][i] = sum(w_basis_func([sol_struct.t, x_nodes[i]],nn_params) .* x)
        end
    end

    # copy internal variables from cache to solution
    sol_struct.internal.x[sol_struct.current_step] .= cache(int).x 
    sol_struct.t = (sol_struct.current_step+1) * timestep(int)
    # println("In the end of update! function, time = ", sol_struct.t)
end

function internal_variables(int::PDEIntegrator{<:TrialNN_PDE_int},problem::PDEProblem)
    local x = cache(int).x
    ntime = Int((problem.tspan[2] - problem.tspan[1]) / problem.tstep)
    xx = (x, ntuple( _ -> zeros(size(x)...), ntime)...)
    previous_params = cache(int).basis_nn_ps
    return (x = xx, previous_params = previous_params)
end

