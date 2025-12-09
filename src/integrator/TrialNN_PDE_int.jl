struct TrialNN_PDE_int{BT<:AbstractPDEBasis,IPMT<:InitialParametersMethod} <: PDEMethod
    basis::BT
    time_quadrature
    RT::Int # Number of quadrature points in time

    spatial_quadrature
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions

    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights

    N_in::Int # Inside the Domain
    x_nodes 
    N_nodes::Int # Number of spatial nodes

    initial_guess_method::IPMT # :LSGD or :GroundTruth

    Nw                 # angular directions
    Nb                 # bias samples

    function TrialNN_PDE_int(trial_NN,;xstep,xspan, RT::Int=6, RX::Int=8, N_in::Int=600, initial_guess_method::IPMT=TrialOGA2D(),
        Nw::Int = 300,Nb::Int = 300) where {IPMT} # 300,300
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

        x_nodes = collect(xspan[1]:xstep:xspan[2])
        N = length(x_nodes)
        new{typeof(trial_NN),typeof(initial_guess_method)}(trial_NN,
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights, N_in, x_nodes,N,initial_guess_method,
            Nw,Nb)
    end
end


default_solver(::TrialNN_PDE_int) = Newton()

struct TrialNN_PDE_intCache{ST,RT,RX,D,NP} <: PDEIntegratorCache{ST,D}
    """
    RT = number of quadrature points in time
    RX = number of quadrature points in space
    D = dimension of output of u,v,w, i.e. scaler value function (D = 1) or vector function
    NP = number of parameters in the expression
    """
    x::Vector{ST}
    W1::Matrix{ST}
    bias1::Vector{ST}

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

    ∂u∂θ_quad_values::Array{ST}
    ∂v∂θ_quad_values::Array{ST}
    ∂w∂θ_quad_values::Array{ST}
    function TrialNN_PDE_intCache{ST,RT,RX,D,NP}() where {ST,RT,RX,D,NP}
        x = zeros(ST, NP) # in ELM, x is just the output layer parameters
        W1 = zeros(ST, NP, 2)
        bias1 = zeros(ST, NP)

        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        ∂u∂θ_quad_values = zeros(ST, D, RT, RX, NP)
        ∂v∂θ_quad_values = zeros(ST, D, RT, RX, NP)
        ∂w∂θ_quad_values = zeros(ST, D, RT, RX, NP)

        new(x,W1,bias1,
            u_quad_values,v_quad_values,w_quad_values,
            ∂L∂U_quad_values,∂L∂V_quad_values,∂L∂W_quad_values,
            ∂u∂θ_quad_values,∂v∂θ_quad_values,∂w∂θ_quad_values,
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

function NN(t, x, W2,W1,bias1,int)
    local activation = int.method.basis.activation_function
    return sum(W2[i] * activation(W1[i,1]*t + W1[i,2]*x + bias1[i]) for i in eachindex(W2))
end

function T1NN_manual(t, x, W2,W1,bias1,int)
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local h = timestep(int)
    local x_domain = b-a

    return (b - x) / x_domain * NN(t, a, W2,W1,bias1,int) +
           (x - a) / x_domain * NN(t, b, W2,W1,bias1,int) +
           (h - h * t) / h * NN(0.0, x, W2,W1,bias1,int)
end

function T2NN_manual(t, x, W2,W1,bias1,int)
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local h = timestep(int)
    local x_domain = b-a

    return (b - x) / x_domain * (h - h * t) / h * NN(0.0, a, W2,W1,bias1,int) +
           (x - a) / x_domain * (h - h * t) / h * NN(0.0, b, W2,W1,bias1,int)
end


function C1(t, x, tn,int, sol)
    local xspan = int.problem.xspan
    local a,b = xspan[1],xspan[2]
    local x_domain = b-a
    local exact_u = int.problem.exact_u
    local h = timestep(int)
    local current_step = sol.current_step
    local W1 = sol.internal.previous_W1
    local bias1 = sol.internal.previous_bias1
    local W2 = sol.internal.previous_W2

    if current_step == 1   
        return (b - x) * exact_u(t, a) / x_domain +
           (x - a) * exact_u(t, b) / x_domain +
           (h - h * t) * exact_u(tn, x) / h
    else
        return (b - x) * exact_u(t, a) / x_domain +(x - a) * exact_u(t, b) / x_domain + (h - h * t) * NN(t, x, W2,W1,bias1,int) / h
                                
    end
end

function C2(t, x, tn,int,sol)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local h = timestep(int)
    local exact_u = int.problem.exact_u

    return (b - x) * (h - h * t) * exact_u(tn, a) / x_domain / h +
           (x - a) * (h - h * t) * exact_u(tn, b) / x_domain / h
end

function u_trial(t, x, W2,W1,bias1,int,sol)
    local current_step = sol.current_step
    local h = timestep(int)
    local tn = (current_step-1)*h
    NN(t,x,W2,W1,bias1,int) - T1NN_manual(t, x, W2,W1,bias1,int) + T2NN_manual(t, x, W2,W1,bias1,int) + C1(t, x, tn,int, sol) - C2(t, x, tn, int, sol)
end

v_trial_zygote(t, x, W2,W1,bias1,int,sol) = Zygote.gradient(tt -> u_trial(tt,x,W2,W1,bias1,int,sol),t)[1]
w_trial_zygote(t, x, W2,W1,bias1,int,sol) = Zygote.gradient(xx -> u_trial(t,xx,W2,W1,bias1,int,sol),x)[1]

v_trial(t, x, W2,W1,bias1,int,sol) = ForwardDiff.derivative(tt -> u_trial(tt,x,W2,W1,bias1,int,sol),t)[1]
w_trial(t, x, W2,W1,bias1,int,sol) = ForwardDiff.derivative(xx -> u_trial(t,xx,W2,W1,bias1,int,sol),x)[1]

∂u∂W2(t, x, W2,W1,bias1,int,sol) = ForwardDiff.gradient(p -> u_trial(t,x,p,W1,bias1,int,sol),W2)
∂v∂W2(t, x, W2,W1,bias1,int,sol) = ForwardDiff.gradient(p -> v_trial(t,x,p,W1,bias1,int,sol),W2)
∂w∂W2(t, x, W2,W1,bias1,int,sol) = ForwardDiff.gradient(p -> w_trial(t,x,p,W1,bias1,int,sol),W2)


function mse_loss(params, tx_in, u_trial,int,sol)
    local current_step = sol.current_step
    local h = timestep(int)
    local tn = (current_step-1)*h
    local exact_u = int.problem.exact_u

    loss = 0.0
    for i in 1:size(tx_in, 2)
        t_samples, x_samples = tx_in[:, i]
        pred = u_trial(t_samples, x_samples, tn, params,int,sol)
        label = exact_u(h .* t_samples, x_samples)
        loss += (pred - label)^2
    end
    return loss / size(tx_in, 2)
end

function prior_initial_guess!(C, sol, int::PDEIntegrator{<:TrialNN_PDE_int{BT,IPMT}}) where {BT,IPMT<:PINN}
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local xspan = int.problem.xspan

    local h = timestep(int)
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local PNN = int.method.basis.sol_network
    local BNN = int.method.basis.basis_network
    local D = int.problem.D
    local N_nodes = int.method.N_nodes
    local x_nodes = int.method.x_nodes
    local current_step = sol.current_step
    local tn = (current_step-1)*h
    local exact_u = int.problem.exact_u
    C.current_step[1] = current_step
    psi_L(x) = (b - x) / x_domain       # lefts
    psi_R(x) = (x - a) / x_domain       # right
    phi_B(t) = (h - t) / h   # bottom
    
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

function prior_initial_guess!(C, sol, int::PDEIntegrator{<:TrialNN_PDE_int{BT,IPMT}}) where {BT,IPMT<:TrialOGA2D}
    local h = timestep(int)
    local Nw = int.method.Nw
    local Nb = int.method.Nb
    local activation = int.method.basis.activation_function
    local K = int.method.basis.NP   
    local a,b = int.problem.xspan[1],int.problem.xspan[2]
    local exact_u = int.problem.exact_u
    local x_domain = b - a
    # local quad_nodes = int.method.grid_matrix
    # local quad_weights = int.method.grid_weights

    # quad_nodes_tuple = reshape(quad_nodes, :, 1)
    # quad_weights = reshape(quad_weights, :, 1)

    # Equidistant Quadrature / sampling grid
    nx = 40
    nt = 20

    xs = range(0.0, 1.0, length=nx)
    ts = range(0.0, 1.0, length=nt)

    # build list of sample coords as 2×N matrix (t; x)
    coords = [ (t,x) for t in ts, x in xs ]   # nt × nx array of tuples
    N = length(coords)
    quad_nodes = zeros(2, N)
    # for i in 1:N
    #     quad_nodes[1, i] = h * quad_nodes_tuple[i][1]
    #     quad_nodes[2, i] = a + (b-a) * quad_nodes_tuple[i][2]
    # end

    for i in 1:N
        quad_nodes[1, i] = coords[i][1]
        quad_nodes[2, i] = coords[i][2]
    end


    # simple uniform quadrature weights (you can switch to Simpson)
    quad_weights = fill(1.0/N, N)
    thetas = range(-π, π, length=Nw+1)
    dirs = [ [cos(θ), sin(θ)] for θ in thetas ]  # length Nw+1

    biases = range(-π, π, length=Nb+1)       # larger bias range works well for sinusoids

    # make dictionary rows (M × 3)
    Arows = Float64[]
    for w in dirs, b in biases
        append!(Arows, [w[1], w[2], b])
    end
    A_mat = reshape(Arows, 3, :)'   # M × 3
    M = size(A_mat,1)

    # build augmented coordinates (for bias): 3 × N
    Xaug = vcat(quad_nodes, ones(1, N))

    # precompute dictionary activations (M×N)
    Φ_raw = activation.(A_mat * Xaug)   # M × N
    # This performs up to `max_iter` outer iterations to account for boundary terms depending on PNN
    selected = Int[]
    B = Matrix{Float64}(undef, N, 0)   # orthonormal basis columns
    coeffs_full = zeros(K)             # coefficients to write into PNN L2
    Wsel = zeros(K, 2)
    Bsel = zeros(K)

    # Build the desired internal PNN output on all quadrature nodes:
    # desired = target + T1NN - T2NN - C1 + C2  (evaluated with current PNN.params)
    desired = zeros(N)
    for i in 1:N
        t = quad_nodes[1,i]; x = quad_nodes[2,i]
        desired[i] = exact_u(h * t, a + (b-a)* x) - u_trial(t, x, coeffs_full,Wsel,Bsel,int,sol)
    end
    @show desired

    # Run OGA (orthogonal matching) on Φ_raw to approximate `desired`
    residual = copy(desired)

    for k = 1:K
        # compute correlations with residual (weighted)
        corrs = zeros(M)
        for i in 1:M
            corrs[i] = abs(sum(Φ_raw[i, :] .* (residual .* quad_weights)))
        end
        idx = argmax(corrs)
        push!(selected, idx)

        # extract raw atom (already normalized) and orthogonalize (Gram-Schmidt)
        φ = copy(Φ_raw[idx, :])

        # append to B
        B = hcat(B, φ)

        # solve least-squares for coefficients in orthonormal basis
        coeffs = B \ desired         # small system k×1 solved implicitly
        # update residual
        residual = desired - B * coeffs

        # store selection params (note A_mat rows correspond to atoms prior to normalization,
        # yet we normalized Φ_raw; we must store original (w,b) for a neuron consistent with A_mat)
        Wsel[k, :] .= A_mat[idx, 1:2]
        Bsel[k] = A_mat[idx, 3]

        coeffs_full[1:k] .= coeffs
        println("k=$k idx=$idx ‖residual‖=$(norm(residual))")
    end

    #     coeffs_full = [  0.6326619468608073, 0.00884977239310545,
    #   0.4315366403167107,
    #  -0.23097264827234718,
    #  -0.8639062057531923,
    #  -0.34446633136732313,
    #  -0.33641779690674506,
    #   0.12898418624582633,
    #   0.34569861714766936,
    #  -0.27813745808117274]

    #     Wsel = [  0.368125      0.929776
    #   0.929776     -0.368125
    #  -0.957319     -0.289032
    #  -1.83697e-16  -1.0
    #  -0.653421      0.756995
    #   0.684547      0.728969
    #   0.368125     -0.929776
    #  -0.770513      0.637424
    #  -0.604599      0.79653
    #   0.570714      0.821149]

    #   Bsel = [  3.141592653589793,
    #   3.141592653589793,
    #   3.141592653589793,
    #   2.9112091923265417,
    #   3.141592653589793,
    #  -0.43982297150257105,
    #   0.41887902047863906,
    #  -0.1466076571675237,
    #  -0.48171087355043496,
    #  -0.25132741228718347]

    for j = 1:K
        C.W1[j, :] .= Wsel[j, :]
        C.bias1[j] = Bsel[j]
        C.x[j] = coeffs_full[j]
    end
    @show length(Set(selected)) == K  # number of unique selected atoms

    target_vec = [exact_u(h*quad_nodes[1,i], a + x_domain * quad_nodes[2,i]) for i in 1:N ]
    approx_vec = [u_trial(quad_nodes[1,i], quad_nodes[2,i], C.x,C.W1,C.bias1,int,sol)  for i in 1:N ]
    err_vec = abs.(target_vec .- approx_vec)
    println("Max abs error after OGA initial guess: ", maximum(err_vec))
    println("OGA initial guess completed.")
    println("Initial guess \n", C.x)
end

initialize_bcs_ics!(sol,int::PDEIntegrator{<:TrialNN_PDE_int}) = nothing

function components!(x::AbstractVector{ST}, sol, int::PDEIntegrator{<:TrialNN_PDE_int}) where {ST}
    local problem_params = int.problem.lagrangian_system.params
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

    local W1 = cache(int).W1
    local bias1 = cache(int).bias1

    t2 = time()
    for d in 1:D
        for i in 1:RT
            for j in 1:RX 
                C.∂u∂θ_quad_values[d, i, j, :] = ∂u∂W2(grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2],x,W1,bias1,int,sol)
                C.∂v∂θ_quad_values[d, i, j, :] = ∂v∂W2(grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2],x,W1,bias1,int,sol)
                C.∂w∂θ_quad_values[d, i, j, :] = ∂w∂W2(grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2],x,W1,bias1,int,sol)
            end
        end
    end
    t3 = time()
    # println("Time for ∂u∂W_quad_values computation: ", t3 - t2)

    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                C.u_quad_values[d, rt, rx] = u_trial(grid_matrix[rt, rx][1], xspan[1] + x_domain* grid_matrix[rt, rx][2],x,W1,bias1,int,sol)
                C.v_quad_values[d, rt, rx] = v_trial_zygote(grid_matrix[rt, rx][1], xspan[1] + x_domain* grid_matrix[rt, rx][2],x,W1,bias1,int,sol)
                C.w_quad_values[d, rt, rx] = w_trial_zygote(grid_matrix[rt, rx][1], xspan[1] + x_domain* grid_matrix[rt, rx][2],x,W1,bias1,int,sol)
            end
        end
    end
    t3 = time()
    # println("Time for u,v,w quad values computation: ", t3 - t2)

    @show C.u_quad_values[1,1,:]
    @show C.v_quad_values[1,1,:]
    @show C.w_quad_values[1,1,:]

    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w
    local quad_x_nodes = int.method.spatial_quadrature.nodes
    local quad_t_nodes = int.method.time_quadrature.nodes
    local h = timestep(int)
    
    @show C.u_quad_values[1,1,:] .- exact_u.(h * quad_t_nodes[1], quad_x_nodes)
    @show (C.v_quad_values[1,1,:] / h) .- exact_v.(h * quad_t_nodes[1], quad_x_nodes)
    @show C.w_quad_values[1,1,:] .- exact_w.(h * quad_t_nodes[1], quad_x_nodes)

    # Compute ∂L/∂θ at quadrature points
    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], problem_params)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], problem_params)
                C.∂L∂W_quad_values[d, i, j] = ∂L∂W[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], problem_params)
            end
        end 
    end
    t4 = time()
    @infiltrate
    # println("Time for ∂L∂U,V,W quad values computation: ", t4 - t3)
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
                        +           timestep(int)  * C.∂L∂W_quad_values[d,rt,rx] * C.∂w∂θ_quad_values[d,rt,rx,p])
                end
            end
            b[p] = -z
        end
    end
    # println("In the end of residual! function, b = ", b)
    @infiltrate
end

function update!(sol_struct, int::PDEIntegrator{<:TrialNN_PDE_int})
    local D = int.problem.D
    local current_step = sol_struct.current_step
    local x_nodes = int.method.x_nodes
    local N_nodes = int.method.N_nodes
    local C = cache(int)
    local W2 = nlsolution(int)
    local W1 = C.W1
    local bias1 = C.bias1

    for d in 1:D
        for i in 1:N_nodes 
            sol_struct.sol.u[current_step][i] = u_trial(1.0,x_nodes[i], W2,W1,bias1,int,sol_struct)
            sol_struct.sol.v[current_step][i] = v_trial_zygote(1.0,x_nodes[i],W2,W1,bias1,int,sol_struct)
            sol_struct.sol.w[current_step][i] = w_trial_zygote(1.0,x_nodes[i],W2,W1,bias1,int,sol_struct)
        end
    end
    
    sol_struct.internal.previous_W1 .= W1
    sol_struct.internal.previous_bias1 .= bias1
    sol_struct.internal.previous_W2 .= W2
    
    # copy internal variables from cache to solution
    sol_struct.t = sol_struct.current_step * timestep(int)
    # println("In the end of update! function, time = ", sol_struct.t)
end

function internal_variables(int::PDEIntegrator{<:TrialNN_PDE_int}, problem::PDEProblem)
    local NP = int.method.basis.NP

    W1 = zeros(NP, 2)
    W2 = zeros(NP)
    bias1 = zeros(NP)
    return (previous_W1 = W1,previous_bias1 = bias1,
        previous_W2 = W2,
        )
end

