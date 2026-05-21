using JLD2
struct ELM_PDE_int{MVT,LT,BT<:AbstractPDEBasis,IPMT<:InitialParametersMethod} <: PDEMethod
    basis::BT

    time_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RT::Int # Number of quadrature points in time

    spatial_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions

    grid_matrix::Matrix{Vector{Float64}} # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights::Matrix{Float64} # Quadrature weights

    t₀_quad::Vector{Vector{Float64}} # quadrature points on t=0 boundary
    t₁_quad::Vector{Vector{Float64}}
    x₀_quad::Vector{Vector{Float64}}
    x₁_quad::Vector{Vector{Float64}}

    t_num_interval::Int
    Nbasis_μ_t::Int
    k_μ_t::Int # order
    μ₀_t::MVT
    μ₁_t::MVT

    x_num_interval::Int
    Nbasis_λ_x::Int
    k_λ_x::Int # order
    λ_x::LT

    mλ_x::Matrix{Float64} # λ_x evaluated at quadrature points
    mμ_t::Matrix{Float64}

    initial_guess_method::IPMT # :LSGD or :GroundTruth

    nepochs::Int
    GD_lr::Float64

    show_status::Bool
    function ELM_PDE_int(basis; RT_per_interval::Int = 4,RX_per_interval::Int = 4,
        xspan::Tuple = (0.,1.0), t_num_interval::Int=10,x_num_interval::Int=10,
        Nbasis_μ_t::Int = 10,k_μ_t::Int = 4,μ::Symbol = :BSplineDirichlet,
        Nbasis_λ_x::Int = 10,k_λ_x::Int = 3,λ::Symbol = :BSplineDirichlet,
        nepochs::Int = 100,GD_lr::Float64 = 1e-5,
        initial_guess_method::IPMT = LSGD(), # ELM()
        show_status = false) where {IPMT,}

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

        # construct quad point at 4 boundary
        t₀_quad = [[0.0, x_quadrature.nodes[i]] for i in 1:RX]
        t₁_quad = [[1.0, x_quadrature.nodes[i]] for i in 1:RX]
        x₀_quad = [[t_quadrature.nodes[i], xspan[1]] for i in 1:RT]
        x₁_quad = [[t_quadrature.nodes[i], xspan[2]] for i in 1:RT]
        # Construct Lagrangian multipliers, defined on [0,1] and need to be scaled carefully when used
        λ_x = Lagrangian_multiplier(λ,Nbasis_λ_x,k_λ_x,xspan[1],xspan[2])
        μ₀_t = Lagrangian_multiplier(μ,Nbasis_μ_t,k_μ_t,0.0,1.0)
        μ₁_t = Lagrangian_multiplier(μ,Nbasis_μ_t,k_μ_t,0.0,1.0)

        mλ_x = zeros(Nbasis_λ_x, RX)
        mμ_t = zeros(Nbasis_μ_t, RT)

        for i in 1:Nbasis_λ_x
            mλ_x[i,:] = λ_x.b[i].(xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes)
        end

        for i in 1:Nbasis_μ_t
            mμ_t[i,:] = μ₀_t.b[i].(t_quadrature.nodes)
        end

        new{typeof(μ₀_t),typeof(λ_x),typeof(basis),typeof(initial_guess_method)}(basis,
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            t₀_quad, t₁_quad, x₀_quad, x₁_quad,
            t_num_interval,Nbasis_μ_t,k_μ_t, μ₀_t, μ₁_t,
            x_num_interval,Nbasis_λ_x,k_λ_x,λ_x,
            mλ_x, mμ_t,
            initial_guess_method,
            nepochs, GD_lr,
            show_status)
    end
end

default_solver(::ELM_PDE_int) = NewtonMethod()

struct ELM_PDE_intCache{ST,RT,RX,D,NP,Nbasis_μ_t,Nbasis_λ_x} <: PDEIntegratorCache{ST,D}
    """
    RT = number of quadrature points in time
    RX = number of quadrature points in space
    D = dimension of output of u,v,w, i.e. scaler value function (D = 1) or vector function
    DX = dimension of input of u,v,w, i.e. number of spatial dimensions
    NP = number of parameters in the expression
    """
    x::Vector{ST}

    u_basis_quad_values::Array{ST} # (NP, RT, RX)
    v_basis_quad_values::Array{ST} # (NP, RT, RX)
    w_basis_quad_values::Array{ST} # (NP, RT, RX)

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

    ut₀_basis_quad_values::Array{ST} # bottom boundary, i.e. t = 0
    ut₁_basis_quad_values::Array{ST} # top boundary, i.e. t = T
    vt₀_basis_quad_values::Array{ST}
    vt₁_basis_quad_values::Array{ST}
    wt₀_basis_quad_values::Array{ST}
    wt₁_basis_quad_values::Array{ST}

    ux₀_basis_quad_values::Array{ST} # left boundary, i.e. x = 0
    ux₁_basis_quad_values::Array{ST} # right boundary, i.e. x = L
    vx₀_basis_quad_values::Array{ST}
    vx₁_basis_quad_values::Array{ST}
    wx₀_basis_quad_values::Array{ST}
    wx₁_basis_quad_values::Array{ST}

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

    ics_ut₀_quad_values::Matrix{ST}
    ics_vt₀_quad_values::Matrix{ST}
    ics_wt₀_quad_values::Matrix{ST}

    bc_ux₀_quad_values::Matrix{ST}
    bc_vx₀_quad_values::Matrix{ST}
    bc_wx₀_quad_values::Matrix{ST}

    bc_ux₁_quad_values::Matrix{ST}
    bc_vx₁_quad_values::Matrix{ST}
    bc_wx₁_quad_values::Matrix{ST}

    λ₀_x_coes::Matrix{ST}
    λ₁_x_coes::Matrix{ST}
    μ₀_t_coes::Matrix{ST}
    μ₁_t_coes::Matrix{ST}

    λ₀_quad_values::Matrix{ST}
    λ₁_quad_values::Matrix{ST}
    μ₀_quad_values::Matrix{ST}
    μ₁_quad_values::Matrix{ST}


    init_condition_t₀::Matrix{ST}
    boundary_condition_x₀::Matrix{ST}
    boundary_condition_x₁::Matrix{ST}

    system_matrix::Array{ST}
    system_rhs::Vector{ST}

    flag_done_initial_guess::Vector{ST}

    function ELM_PDE_intCache{ST,RT,RX,D,NP,Nbasis_μ_t,Nbasis_λ_x}() where {ST,RT,RX,D,NP,Nbasis_μ_t,Nbasis_λ_x}
        x = zeros(ST, NP + D * Nbasis_λ_x + 2 * D * Nbasis_μ_t) # params, λ₁_x_coes,μ₀_t_coes,μ₁_t_coes

        u_basis_quad_values = zeros(ST, D, NP, RT, RX)
        v_basis_quad_values = zeros(ST, D, NP, RT, RX)
        w_basis_quad_values = zeros(ST, D, NP, RT, RX)

        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        ut₀_basis_quad_values = zeros(ST, D, RX, NP) # bottom boundary, i.e. t = 0
        ut₁_basis_quad_values = zeros(ST, D, RX, NP) # top boundary, i.e. t = T
        vt₀_basis_quad_values = zeros(ST, D, RX, NP)
        vt₁_basis_quad_values = zeros(ST, D, RX, NP)
        wt₀_basis_quad_values = zeros(ST, D, RX, NP)
        wt₁_basis_quad_values = zeros(ST, D, RX, NP)

        ux₀_basis_quad_values = zeros(ST, D, RT, NP) # left boundary, i.e. x = 0
        ux₁_basis_quad_values = zeros(ST, D, RT, NP) # right boundary, i.e. x = L
        vx₀_basis_quad_values = zeros(ST, D, RT, NP)
        vx₁_basis_quad_values = zeros(ST, D, RT, NP)
        wx₀_basis_quad_values = zeros(ST, D, RT, NP)
        wx₁_basis_quad_values = zeros(ST, D, RT, NP)

        ut₀_quad_values = zeros(ST, D, RX)
        ut₁_quad_values = zeros(ST, D, RX)
        vt₀_quad_values = zeros(ST, D, RX)
        vt₁_quad_values = zeros(ST, D, RX)
        wt₀_quad_values = zeros(ST, D, RX)
        wt₁_quad_values = zeros(ST, D, RX)

        ux₀_quad_values = zeros(ST, D, RT)
        ux₁_quad_values = zeros(ST, D, RT)
        vx₀_quad_values = zeros(ST, D, RT)
        vx₁_quad_values = zeros(ST, D, RT)
        wx₀_quad_values = zeros(ST, D, RT)
        wx₁_quad_values = zeros(ST, D, RT)

        ics_ut₀_quad_values = zeros(ST, D, RX)
        ics_vt₀_quad_values = zeros(ST, D, RX)
        ics_wt₀_quad_values = zeros(ST, D, RX)

        bc_ux₀_quad_values = zeros(ST, D, RT)
        bc_vx₀_quad_values = zeros(ST, D, RT)
        bc_wx₀_quad_values = zeros(ST, D, RT)

        bc_ux₁_quad_values = zeros(ST, D, RT)
        bc_vx₁_quad_values = zeros(ST, D, RT)
        bc_wx₁_quad_values = zeros(ST, D, RT)

        λ₀_x_coes = zeros(ST, D, Nbasis_λ_x)
        λ₁_x_coes = zeros(ST, D, Nbasis_λ_x)
        μ₀_t_coes = zeros(ST, D, Nbasis_μ_t)
        μ₁_t_coes = zeros(ST, D, Nbasis_μ_t)

        λ₀_quad_values = zeros(ST, D, RX)
        λ₁_quad_values = zeros(ST, D, RX)
        μ₀_quad_values = zeros(ST, D, RT)
        μ₁_quad_values = zeros(ST, D, RT)

        init_condition_t₀ = zeros(ST, D, RX)
        boundary_condition_x₀ = zeros(ST, D, RT)
        boundary_condition_x₁ = zeros(ST, D, RT)

        system_matrix = zeros(ST, RT * RX +  RX + 2* RT, NP)
        system_rhs = zeros(ST, RT * RX +  RX + 2* RT)

        flag_done_initial_guess = zeros(ST, 1)

        new(x,
            u_basis_quad_values, v_basis_quad_values, w_basis_quad_values,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ut₀_basis_quad_values, ut₁_basis_quad_values, vt₀_basis_quad_values, vt₁_basis_quad_values, wt₀_basis_quad_values, wt₁_basis_quad_values,
            ux₀_basis_quad_values, ux₁_basis_quad_values, vx₀_basis_quad_values, vx₁_basis_quad_values, wx₀_basis_quad_values, wx₁_basis_quad_values,
            ut₀_quad_values, ut₁_quad_values, vt₀_quad_values, vt₁_quad_values, wt₀_quad_values, wt₁_quad_values,
            ux₀_quad_values, ux₁_quad_values, vx₀_quad_values, vx₁_quad_values, wx₀_quad_values, wx₁_quad_values,
            ics_ut₀_quad_values, ics_vt₀_quad_values, ics_wt₀_quad_values,
            bc_ux₀_quad_values, bc_vx₀_quad_values, bc_wx₀_quad_values,
            bc_ux₁_quad_values, bc_vx₁_quad_values, bc_wx₁_quad_values,
            λ₀_x_coes, λ₁_x_coes, μ₀_t_coes, μ₁_t_coes,
            λ₀_quad_values, λ₁_quad_values, μ₀_quad_values, μ₁_quad_values,
            init_condition_t₀, boundary_condition_x₀, boundary_condition_x₁,
            system_matrix, system_rhs, flag_done_initial_guess)
    end
end

nlsolution(cache::ELM_PDE_intCache) = cache.x

function Cache{ST}(problem::LPDEProblem, int::ELM_PDE_int; kwargs...) where {ST}
    ELM_PDE_intCache{ST,int.RT,int.RX,problem.D,int.basis.NP,int.Nbasis_μ_t,int.Nbasis_λ_x}(; kwargs...)
end

@inline CacheType(ST, problem::LPDEProblem, int::ELM_PDE_int) = ELM_PDE_intCache{ST,int.RT,int.RX,problem.D,int.basis.NP,int.Nbasis_μ_t,int.Nbasis_λ_x}

function prior_initial_guess!(C, sol, int::PDEIntegrator{<:ELM_PDE_int{MVT,LT,BT,IPMT}}) where {MVT,LT,BT,IPMT<:ELM}
    # Currently Unsupported!!!
    local D = int.problem.D
    local RT = int.method.RT
    local RX = int.method.RX
    local lsq_assemble = int.problem.least_squares_assemble
    local grid_matrix = int.method.grid_matrix
    local nn_params = int.method.basis.u_basis.params
    local v_basis_func = int.method.basis.v_basis
    local w_basis_func = int.method.basis.w_basis
    local u_basis_func = int.method.basis.u_basis
    local t₀_quad = int.method.t₀_quad
    local t₁_quad = int.method.t₁_quad
    local x₀_quad = int.method.x₀_quad
    local x₁_quad = int.method.x₁_quad


    for d in 1:D
        for rx in 1:RX
            C.ut₀_basis_quad_values[d,rx, :] = u_basis_func(t₀_quad[rx], nn_params)
            C.ut₁_basis_quad_values[d,rx, :] = u_basis_func(t₁_quad[rx], nn_params)
            C.vt₀_basis_quad_values[d,rx, :] = v_basis_func(t₀_quad[rx], nn_params)
            C.vt₁_basis_quad_values[d,rx, :] = v_basis_func(t₁_quad[rx], nn_params)
            C.wt₀_basis_quad_values[d,rx, :] = w_basis_func(t₀_quad[rx], nn_params)
            C.wt₁_basis_quad_values[d,rx, :] = w_basis_func(t₁_quad[rx], nn_params)
        end

        for rt in 1:RT
            C.ux₀_basis_quad_values[d,rt, :] = u_basis_func(x₀_quad, nn_params)
            C.ux₁_basis_quad_values[d,rt, :] = u_basis_func(x₁_quad, nn_params)
            C.vx₀_basis_quad_values[d,rt, :] = v_basis_func(x₀_quad, nn_params)
            C.vx₁_basis_quad_values[d,rt, :] = v_basis_func(x₁_quad, nn_params)
            C.wx₀_basis_quad_values[d,rt, :] = w_basis_func(x₀_quad, nn_params)
            C.wx₁_basis_quad_values[d,rt, :] = w_basis_func(x₁_quad, nn_params)
        end

        for rt in 1:RT
            for rx in 1:RX
                @views C.u_basis_quad_values[d, :, rt, rx] = u_basis_func(grid_matrix[rt, rx], nn_params)
                @views C.v_basis_quad_values[d, :, rt, rx] = v_basis_func(grid_matrix[rt, rx], nn_params)
                @views C.w_basis_quad_values[d, :, rt, rx] = w_basis_func(grid_matrix[rt, rx], nn_params)
            end
        end

    end

    lsq_assemble(C,int,sol)
    # function elm_lsq!(du,x,p)
    #     du= cache_.system_matrix * x .- cache_.system_rhs
    # end

    # prob = NonlinearLeastSquaresProblem(
    #     NonlinearFunction(elm_lsq!, resid_prototype = zeros(3)), zeros(NP), cache_)
    # ls_sol = solve(prob)
    # @show ls_sol
    # C.x[:] = ls_sol.u
    # println("After least square, the parameters are: ", C.x)

end

function prior_initial_guess!(C, sol, int::PDEIntegrator{<:ELM_PDE_int{MVT,LT,BT,IPMT}}) where {MVT,LT,BT,IPMT<:LSGD}
    local NP = int.method.basis.NP
    local RT = int.method.RT
    local RX = int.method.RX
    local xspan = int.problem.xspan
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local x_domain = xspan[2] - xspan[1]
    local NN = int.method.basis.network_arch
    local PNN = int.method.basis.u
    local nepochs = int.method.nepochs
    local exact_u = int.problem.exact_u
    local h = timestep(int)
    local show_status = int.method.show_status
    local GD_lr = int.method.GD_lr
    local D = int.problem.D
    local u_basis_func = int.method.basis.u_basis
    local v_basis_func = int.method.basis.v_basis
    local w_basis_func = int.method.basis.w_basis

    local nn_params = u_basis_func.params
    network_inputs, _ = construct_quadrature_grid_with_boundary([RT, RX])
    network_inputs[2,:] .= xspan[1] .+ x_domain .* network_inputs[2, :]
    labels = exact_u.(sol.t .- h .+ h .* network_inputs[1, :],network_inputs[2, :])
    labels = reshape(labels, 1, :)

    tem_ps = PNN.params[keys(PNN.params)[1:end-1]]
    opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.GradientOptimizer(GD_lr), tem_ps)
    err = 0
    λ = GeometricMachineLearning.GlobalSection(tem_ps)

    ls_err = zeros(nepochs)
    gd_err = zeros(nepochs)
    ls_max_err = zeros(nepochs)
    gd_max_err = zeros(nepochs)

    for ep in 1:nepochs
        tem_ps = PNN.params[keys(PNN.params)[1:end-1]]
        gs = Zygote.gradient(p -> lsgd_loss(network_inputs, labels, PNN, p), PNN.params)[1]
        tem_gs = gs[keys(gs)[1:end-1]]
        GeometricMachineLearning.optimization_step!(opt, λ, tem_ps, tem_gs)

        if show_status
            gd_err[ep] = lsgd_loss(network_inputs, labels, PNN, PNN.params)
            print("\n loss after gradient: $gd_err[ep] by $ep epochs")

            NN_output = PNN(network_inputs, PNN.params)
            gd_max_err[ep] = maximum(abs.(labels .- NN_output))
            println("max error :",gd_max_err[ep])
        end

        Φ = AbstractNeuralNetworks.Chain(PNN.model.layers[1:end-1]...)(network_inputs, tem_ps)
        PNN.params[keys(PNN.params)[end]].W[:] = (Φ' \ labels')'

        ls_err[ep] = lsgd_loss(network_inputs, labels, PNN, PNN.params)
        print("\n loss after least square: $(ls_err[ep]) by $ep epochs")

        NN_output = PNN(network_inputs, PNN.params)
        ls_max_err[ep] = maximum(abs.(labels .- NN_output))

        if ep == nepochs && show_status
            println("max error :",ls_max_err[ep])
            record_results = Dict()
            record_results["max_error"] = gd_max_err
            record_results["gd_err"] = gd_err
            record_results["ls_max_error"] = ls_max_err
            record_results["ls_err"] = ls_err
            record_results["PNN_params"] = PNN.params
            JLD2.save("LSGD_initial_guess_results.jld2", record_results)
            print("Results saved!!!")
            @show PNN.params[keys(PNN.params)[end]].W[:]
        end

        # if ls_max_err[ep] <1e-7
        #     print("\n final max error : $(ls_max_err[ep]) by $ep epochs")
        #     break
        # elseif ep == nepochs
        #     print("\n final loss: $err by $ep epochs")
        # end
    end

    # if show_status
    #     pic = Plots.plot(
    #         Plots.plot(1:nepochs, gd_err, title="Gradient Descent Loss"),
    #         Plots.plot(1:nepochs, ls_err, title="Least Square Loss"),
    #         Plots.plot(1:nepochs, gd_max_err, title="Gradient Descent Max Error"),
    #         Plots.plot(1:nepochs, ls_max_err, title="Least Square Max Error"),
    #         layout=(2,2), size=(1000, 800)
    #     )
    #     Plots.savefig(pic, "logs/elm_training_loss.pdf")

    #     x_ls = collect(xspan[1]:0.01:xspan[2])
    #     t_ls = collect(0:0.01:1.0)

    #     u_ls = [lpde.exact_u(h*t, x) for x in x_ls, t in t_ls]
    #     u_pred_ls = [PNN([t, x], PNN.params)[1] for x in x_ls, t in t_ls]
    #     pic2 = Plots.plot(
    #         Plots.surface(x_ls, t_ls, u_ls', title="Exact Solution", xlabel="x", ylabel="t", zlabel="u"),
    #         Plots.surface(x_ls, t_ls, u_pred_ls', title="Predicted Solution", xlabel="x", ylabel="t", zlabel="u"),
    #         Plots.surface(x_ls, t_ls, abs.(u_ls .- u_pred_ls)', title="Absolute Error", xlabel="x", ylabel="t", zlabel="|u - u_pred|"),
    #         layout=(1,3), size=(1000, 400)
    #     )
    #     Plots.savefig(pic2, "logs/elm_training_solution.pdf")
    # end
    # copy the parameters to the cache
    C.x[1:NP] = PNN.params[keys(PNN.params)[end]].W[:]

    #copy PNN parameters into u,v,w basis
    for (name, layer) in zip(keys(u_basis_func.params), values(u_basis_func.params))
        layer.W[:], layer.b[:] = PNN.params[name].W[:], PNN.params[name].b[:]
    end

    # precompute basis function values at quadrature points
    for d in 1:D
        for rx in 1:RX
            C.ut₀_basis_quad_values[d,rx, :] = u_basis_func([0.0, xspan[1] + x_domain * x_quad_nodes[rx]], nn_params)
            C.ut₁_basis_quad_values[d,rx, :] = u_basis_func([1.0, xspan[1] + x_domain * x_quad_nodes[rx]], nn_params)
            C.vt₀_basis_quad_values[d,rx, :] = v_basis_func([0.0, xspan[1] + x_domain * x_quad_nodes[rx]], nn_params)
            C.vt₁_basis_quad_values[d,rx, :] = v_basis_func([1.0, xspan[1] + x_domain * x_quad_nodes[rx]], nn_params)
            C.wt₀_basis_quad_values[d,rx, :] = w_basis_func([0.0, xspan[1] + x_domain * x_quad_nodes[rx]], nn_params)
            C.wt₁_basis_quad_values[d,rx, :] = w_basis_func([1.0, xspan[1] + x_domain * x_quad_nodes[rx]], nn_params)
        end

        for rt in 1:RT
            C.ux₀_basis_quad_values[d,rt, :] = u_basis_func([t_quad_nodes[rt], xspan[1]], nn_params)
            C.ux₁_basis_quad_values[d,rt, :] = u_basis_func([t_quad_nodes[rt], xspan[2]], nn_params)
            C.vx₀_basis_quad_values[d,rt, :] = v_basis_func([t_quad_nodes[rt], xspan[1]], nn_params)
            C.vx₁_basis_quad_values[d,rt, :] = v_basis_func([t_quad_nodes[rt], xspan[2]], nn_params)
            C.wx₀_basis_quad_values[d,rt, :] = w_basis_func([t_quad_nodes[rt], xspan[1]], nn_params)
            C.wx₁_basis_quad_values[d,rt, :] = w_basis_func([t_quad_nodes[rt], xspan[2]], nn_params)
        end

        for rt in 1:RT
            for rx in 1:RX
                C.u_basis_quad_values[d, :, rt, rx] = u_basis_func([t_quad_nodes[rt], xspan[1] + x_domain * x_quad_nodes[rx]], nn_params)
                C.v_basis_quad_values[d, :, rt, rx] = v_basis_func([t_quad_nodes[rt], xspan[1] + x_domain * x_quad_nodes[rx]], nn_params)
                C.w_basis_quad_values[d, :, rt, rx] = w_basis_func([t_quad_nodes[rt], xspan[1] + x_domain * x_quad_nodes[rx]], nn_params)
            end
        end
        @show nn_params.L1.b[:]
    end

end

copy_internal_variables!(C::ELM_PDE_intCache, solstep::SolutionStep) = nothing

function post_initial_guess!(C, sol, int::PDEIntegrator{<:ELM_PDE_int}, int_method::ELM_PDE_int{MVT,LT,BT,IPMT}) where {MVT<:BSplineDirichlet,LT<:BSplineDirichlet,BT,IPMT}
    local Nbasis_λ_x = int_method.Nbasis_λ_x
    local Nbasis_μ_t = int_method.Nbasis_μ_t
    local D = int.problem.D
    local lag_sys = int.problem.lagrangian_system.functions
    local params = int.problem.lagrangian_system.params
    local NP = int_method.basis.NP
    local RX = int.method.RX
    local RT = int.method.RT
    local mλ_x = int_method.mλ_x
    local mμ_t = int_method.mμ_t

    for d in 1:D
        tem_t₁_∂L∂V = zeros(RX)
        tem_x₀_∂L∂W = zeros(RT)
        tem_x₁_∂L∂W = zeros(RT)

        for rx in 1:RX
            tem_t₁_∂L∂V[rx] = lag_sys.∂L∂V[d](C.ut₁_quad_values[d,rx], C.vt₁_quad_values[d,rx], C.wt₁_quad_values[d,rx], params)
        end
        λ_x_tem = mλ_x'\tem_t₁_∂L∂V

        for rx in 1:Nbasis_λ_x
            C.x[NP + (d - 1) * Nbasis_λ_x + rx] = λ_x_tem[rx]
        end

        for rt in 1:RT
            tem_x₀_∂L∂W[rt] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,rt], C.vx₀_quad_values[d,rt], C.wx₀_quad_values[d,rt], params)
            tem_x₁_∂L∂W[rt] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,rt], C.vx₁_quad_values[d,rt], C.wx₁_quad_values[d,rt], params)
        end
        μ₀_t_tem = mμ_t'\tem_x₀_∂L∂W
        μ₁_t_tem = mμ_t'\tem_x₁_∂L∂W

        for rt in 1:Nbasis_μ_t
            C.x[NP + D * Nbasis_λ_x + (d - 1) * Nbasis_μ_t + rt] = μ₀_t_tem[rt]
            C.x[NP + D * Nbasis_λ_x + + D * Nbasis_μ_t + (d - 1) * Nbasis_μ_t + rt] = μ₁_t_tem[rt]
        end
    end
    C.flag_done_initial_guess[1] = 1.0

    # @infiltrate
end

function post_initial_guess!(C, sol, int::PDEIntegrator{<:ELM_PDE_int}, int_method::ELM_PDE_int{MVT,LT,BT,IPMT}) where {MVT<:Lagrange,LT<:Lagrange,BT,IPMT}
    local NP = int_method.basis.NP
    local D = int.problem.D
    local lag_sys = int.problem.lagrangian_system.functions
    local params = int.problem.lagrangian_system.params
    local u = [int.method.basis.u]
    local v = [int.method.basis.v]
    local w = [int.method.basis.w]
    local Nbasis_λ_x = int_method.Nbasis_λ_x
    local Nbasis_μ_t = int_method.Nbasis_μ_t
    local xspan = int.problem.xspan
    local λ_x = int_method.λ_x
    local μ₀_t = int_method.μ₀_t
    local show_status = int_method.show_status
    local h = timestep(int)
    local nn_params = u_basis_func.params

    ut₀_quad_values_tem = zeros(Nbasis_λ_x)
    ut₁_quad_values_tem = zeros(Nbasis_λ_x)
    vt₀_quad_values_tem = zeros(Nbasis_λ_x)
    vt₁_quad_values_tem = zeros(Nbasis_λ_x)
    wt₀_quad_values_tem = zeros(Nbasis_λ_x)
    wt₁_quad_values_tem = zeros(Nbasis_λ_x)

    ux₀_quad_values_tem = zeros(Nbasis_μ_t)
    ux₁_quad_values_tem = zeros(Nbasis_μ_t)
    vx₀_quad_values_tem = zeros(Nbasis_μ_t)
    vx₁_quad_values_tem = zeros(Nbasis_μ_t)
    wx₀_quad_values_tem = zeros(Nbasis_μ_t)
    wx₁_quad_values_tem = zeros(Nbasis_μ_t)

    for d in 1:D
        for rx in 1:Nbasis_λ_x
            xx = λ_x.x[rx]
            ut₀_quad_values_tem[rx] = sum(u_basis_func([0.0, xx], nn_params) .* C.x[1:NP])
            ut₁_quad_values_tem[rx] = sum(u_basis_func([1.0, xx], nn_params) .* C.x[1:NP])
            vt₀_quad_values_tem[rx] = sum(v_basis_func([0.0, xx], nn_params) .* C.x[1:NP]) / h
            vt₁_quad_values_tem[rx] = sum(v_basis_func([1.0, xx], nn_params) .* C.x[1:NP]) / h
            wt₀_quad_values_tem[rx] = sum(w_basis_func([0.0, xx], nn_params) .* C.x[1:NP])
            wt₁_quad_values_tem[rx] = sum(w_basis_func([1.0, xx], nn_params) .* C.x[1:NP])
        end

        for rt in 1:Nbasis_μ_t
            tt = μ₀_t.x[rt]
            ux₀_quad_values_tem[rt] = sum(u_basis_func([tt, xspan[1]], nn_params) .* C.x[1:NP])
            ux₁_quad_values_tem[rt] = sum(u_basis_func([tt, xspan[2]], nn_params) .* C.x[1:NP])
            vx₀_quad_values_tem[rt] = sum(v_basis_func([tt, xspan[1]], nn_params) .* C.x[1:NP]) / h
            vx₁_quad_values_tem[rt] = sum(v_basis_func([tt, xspan[2]], nn_params) .* C.x[1:NP]) / h
            wx₀_quad_values_tem[rt] = sum(w_basis_func([tt, xspan[1]], nn_params) .* C.x[1:NP])
            wx₁_quad_values_tem[rt] = sum(w_basis_func([tt, xspan[2]], nn_params) .* C.x[1:NP])
        end
    end
    for d in 1:D
        for rx in 1:Nbasis_λ_x
            C.x[NP+(d-1)*Nbasis_λ_x+rx] = lag_sys.∂L∂V[d](ut₁_quad_values_tem[rx], vt₁_quad_values_tem[rx], wt₁_quad_values_tem[rx], params)
        end

        for rt in 1:Nbasis_μ_t
            C.x[NP+D*Nbasis_λ_x+(d-1)*Nbasis_μ_t+rt] = lag_sys.∂L∂W[d](ux₀_quad_values_tem[rt], vx₀_quad_values_tem[rt], wx₀_quad_values_tem[rt], params)
            C.x[NP+D*Nbasis_λ_x++D*Nbasis_μ_t+(d-1)*Nbasis_μ_t+rt] = lag_sys.∂L∂W[d](ux₁_quad_values_tem[rt], vx₁_quad_values_tem[rt], wx₁_quad_values_tem[rt], params)
        end
    end
    C.flag_done_initial_guess[1] = 1.0

    if show_status
        local exact_u = int.problem.exact_u
        local exact_v = int.problem.exact_v
        local exact_w = int.problem.exact_w
        ut₀_truth_quad = zeros(Nbasis_λ_x)
        ut₁_truth_quad = zeros(Nbasis_λ_x)
        vt₁_truth_quad = zeros(Nbasis_λ_x)
        vt₀_truth_quad = zeros(Nbasis_λ_x)
        wt₀_truth_quad = zeros(Nbasis_λ_x)
        wt₁_truth_quad = zeros(Nbasis_λ_x)


        ux₀_truth_quad = zeros(Nbasis_μ_t)
        ux₁_truth_quad = zeros(Nbasis_μ_t)
        vx₀_truth_quad = zeros(Nbasis_μ_t)
        vx₁_truth_quad = zeros(Nbasis_μ_t)
        wx₀_truth_quad = zeros(Nbasis_μ_t)
        wx₁_truth_quad = zeros(Nbasis_μ_t)

        for rx in 1:Nbasis_λ_x
            xx = λ_x.x[rx]
            ut₀_truth_quad[rx] = exact_u.(sol.t - timestep(int), xx)
            ut₁_truth_quad[rx] = exact_u.(sol.t, xx)
            vt₀_truth_quad[rx] = exact_v.(sol.t - timestep(int), xx)
            vt₁_truth_quad[rx] = exact_v.(sol.t, xx)
            wt₀_truth_quad[rx] = exact_w.(sol.t - timestep(int), xx)
            wt₁_truth_quad[rx] = exact_w.(sol.t, xx)
        end

        for rt in 1:Nbasis_μ_t
            tt = μ₀_t.x[rt]
            ux₀_truth_quad[rt] = exact_u.(sol.t - timestep(int) + timestep(int) * tt, xspan[1])
            ux₁_truth_quad[rt] = exact_u.(sol.t - timestep(int) + timestep(int) * tt, xspan[2])
            vx₀_truth_quad[rt] = exact_v.(sol.t - timestep(int) + timestep(int) * tt, xspan[1])
            vx₁_truth_quad[rt] = exact_v.(sol.t - timestep(int) + timestep(int) * tt, xspan[2])
            wx₀_truth_quad[rt] = exact_w.(sol.t - timestep(int) + timestep(int) * tt, xspan[1])
            wx₁_truth_quad[rt] = exact_w.(sol.t - timestep(int) + timestep(int) * tt, xspan[2])
        end

        @show maximum(abs.(ut₀_quad_values_tem .- ut₀_truth_quad))
        @show maximum(abs.(vt₀_quad_values_tem .- vt₀_truth_quad))
        @show maximum(abs.(wt₀_quad_values_tem .- wt₀_truth_quad))
        @show maximum(abs.(ut₁_quad_values_tem .- ut₁_truth_quad))
        @show maximum(abs.(vt₁_quad_values_tem .- vt₁_truth_quad))
        @show maximum(abs.(wt₁_quad_values_tem .- wt₁_truth_quad))

        @show maximum(abs.(ux₀_quad_values_tem .- ux₀_truth_quad))
        @show maximum(abs.(vx₀_quad_values_tem .- vx₀_truth_quad))
        @show maximum(abs.(wx₀_quad_values_tem .- wx₀_truth_quad))
        @show maximum(abs.(ux₁_quad_values_tem .- ux₁_truth_quad))
        @show maximum(abs.(vx₁_quad_values_tem .- vx₁_truth_quad))
        @show maximum(abs.(wx₁_quad_values_tem .- wx₁_truth_quad))
    end
    # @infiltrate

end

function components!(x::AbstractVector{ST}, sol, params, int::PDEIntegrator{<:ELM_PDE_int}) where {ST}
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
    local C_float = cache(int)
    local h = timestep(int)
    local NP = int.method.basis.NP
    local Nbasis_λ_x = int.method.Nbasis_λ_x
    local Nbasis_μ_t = int.method.Nbasis_μ_t
    local mλ_x = int.method.mλ_x
    local mμ_t = int.method.mμ_t
    local lag_params = int.problem.lagrangian_system.params
    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w
    local show_status = int.method.show_status

    coeffs = x[1:NP]

    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                C.u_quad_values[d, rt, rx] = sum(C_float.u_basis_quad_values[d, :, rt,rx] .* coeffs)
                C.v_quad_values[d, rt, rx] = sum(C_float.v_basis_quad_values[d, :, rt,rx] .* coeffs) / h
                C.w_quad_values[d, rt, rx] = sum(C_float.w_basis_quad_values[d, :, rt,rx] .* coeffs)
            end
        end
    end

    for d in 1:D
        for rx in 1:RX
            C.ut₀_quad_values[d,rx] = sum(C_float.ut₀_basis_quad_values[d,rx, :] .* coeffs)
            C.ut₁_quad_values[d,rx] = sum(C_float.ut₁_basis_quad_values[d,rx, :] .* coeffs)
            C.vt₀_quad_values[d,rx] = sum(C_float.vt₀_basis_quad_values[d,rx, :] .* coeffs) / h
            C.vt₁_quad_values[d,rx] = sum(C_float.vt₁_basis_quad_values[d,rx, :] .* coeffs) / h
            C.wt₀_quad_values[d,rx] = sum(C_float.wt₀_basis_quad_values[d,rx, :] .* coeffs)
            C.wt₁_quad_values[d,rx] = sum(C_float.wt₁_basis_quad_values[d,rx, :] .* coeffs)
        end

        for rt in 1:RT
            C.ux₀_quad_values[d,rt] =  sum(C_float.ux₀_basis_quad_values[d,rt, :] .* coeffs)
            C.ux₁_quad_values[d,rt] =  sum(C_float.ux₁_basis_quad_values[d,rt, :] .* coeffs)
            C.vx₀_quad_values[d,rt] =  sum(C_float.vx₀_basis_quad_values[d,rt, :] .* coeffs) / h
            C.vx₁_quad_values[d,rt] =  sum(C_float.vx₁_basis_quad_values[d,rt, :] .* coeffs) / h
            C.wx₀_quad_values[d,rt] =  sum(C_float.wx₀_basis_quad_values[d,rt, :] .* coeffs)
            C.wx₁_quad_values[d,rt] =  sum(C_float.wx₁_basis_quad_values[d,rt, :] .* coeffs)
        end
    end


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


    cache(int).flag_done_initial_guess[1] == 0.0 ? post_initial_guess!(cache(int), sol, int, int.method) : nothing

    for d in 1:D
        @views C.λ₁_x_coes[d, :] = x[NP+1:NP+Nbasis_λ_x]
        @views C.μ₀_t_coes[d, :] = x[NP+Nbasis_λ_x+1:NP+Nbasis_λ_x+Nbasis_μ_t]
        @views C.μ₁_t_coes[d, :] = x[NP+Nbasis_λ_x+Nbasis_μ_t+1:NP+Nbasis_λ_x+2*Nbasis_μ_t]
    end

    for d in 1:D
        for rx in 1:RX
            C.λ₀_quad_values[d, rx] = ∂L∂V[d](C.ics_ut₀_quad_values[d, rx], C.ics_vt₀_quad_values[d, rx], C.ics_wt₀_quad_values[d, rx], lag_params)
            C.λ₁_quad_values[d, rx] = sum(C.λ₁_x_coes[d, :] .* mλ_x[:, rx])
        end

        for rt in 1:RT
            C.μ₀_quad_values[d, rt] = sum(C.μ₀_t_coes[d, :] .* mμ_t[:, rt])
            C.μ₁_quad_values[d, rt] = sum(C.μ₁_t_coes[d, :] .* mμ_t[:, rt])
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

        # boundary values at quadrature points
        ut₀_quad_values_truth = similar(C.ut₀_quad_values)
        ut₁_quad_values_truth = similar(C.ut₁_quad_values)
        vt₀_quad_values_truth = similar(C.vt₀_quad_values)
        vt₁_quad_values_truth = similar(C.vt₁_quad_values)
        wt₀_quad_values_truth = similar(C.wt₀_quad_values)
        wt₁_quad_values_truth = similar(C.wt₁_quad_values)

        ux₀_quad_values_truth = similar(C.ux₀_quad_values)
        ux₁_quad_values_truth = similar(C.ux₁_quad_values)
        vx₀_quad_values_truth = similar(C.vx₀_quad_values)
        vx₁_quad_values_truth = similar(C.vx₁_quad_values)
        wx₀_quad_values_truth = similar(C.wx₀_quad_values)
        wx₁_quad_values_truth = similar(C.wx₁_quad_values)

        for d in 1:D
            for j in 1:RX
                ut₀_quad_values_truth[d,j] = exact_u.(sol.t - timestep(int), xspan[1] .+ x_domain .* x_quad_nodes[j])
                ut₁_quad_values_truth[d,j] = exact_u.(sol.t, xspan[1] .+ x_domain .* x_quad_nodes[j])
                vt₀_quad_values_truth[d,j] = exact_v.(sol.t - timestep(int), xspan[1] .+ x_domain .* x_quad_nodes[j])
                vt₁_quad_values_truth[d,j] = exact_v.(sol.t, xspan[1] .+ x_domain .* x_quad_nodes[j])
                wt₀_quad_values_truth[d,j] = exact_w.(sol.t - timestep(int), xspan[1] .+ x_domain .* x_quad_nodes[j])
                wt₁_quad_values_truth[d,j] = exact_w.(sol.t, xspan[1] .+ x_domain .* x_quad_nodes[j])
            end

            for i in 1:RT
                ux₀_quad_values_truth[d,i] = exact_u.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[1])
                ux₁_quad_values_truth[d,i] = exact_u.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[2])
                vx₀_quad_values_truth[d,i] = exact_v.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[1])
                vx₁_quad_values_truth[d,i] = exact_v.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[2])
                wx₀_quad_values_truth[d,i] = exact_w.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[1])
                wx₁_quad_values_truth[d,i] = exact_w.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[2])
            end
        end

        @show maximum(abs.(C.ut₀_quad_values .- ut₀_quad_values_truth))
        @show maximum(abs.(C.ut₁_quad_values .- ut₁_quad_values_truth))

        @show maximum(abs.(C.vt₀_quad_values .- vt₀_quad_values_truth))
        @show maximum(abs.(C.vt₁_quad_values .- vt₁_quad_values_truth))

        @show maximum(abs.(C.wt₀_quad_values .- wt₀_quad_values_truth))
        @show maximum(abs.(C.wt₁_quad_values .- wt₁_quad_values_truth))

        @show maximum(abs.(C.ux₀_quad_values .- ux₀_quad_values_truth))
        @show maximum(abs.(C.ux₁_quad_values .- ux₁_quad_values_truth))

        @show maximum(abs.(C.vx₀_quad_values .- vx₀_quad_values_truth))
        @show maximum(abs.(C.vx₁_quad_values .- vx₁_quad_values_truth))

        @show maximum(abs.(C.wx₀_quad_values .- wx₀_quad_values_truth))
        @show maximum(abs.(C.wx₁_quad_values .- wx₁_quad_values_truth))

        @show maximum(abs.(C.λ₁_quad_values .- vt₁_quad_values_truth))
        @show maximum(abs.(0.25 .* C.bc_wx₀_quad_values .+ C.μ₀_quad_values))
        @show maximum(abs.(0.25 .* C.bc_wx₁_quad_values .+ C.μ₁_quad_values))
    end
    # @infiltrate
end

function residual!(b::Vector{ST}, sol, params, int::PDEIntegrator{<:ELM_PDE_int}) where {ST,}
    local D = int.problem.D
    local RT = int.method.RT
    local RX = int.method.RX
    local NP = int.method.basis.NP
    local quad_b = int.method.grid_weights
    local C = cache(int, ST)
    local C_float = cache(int)

    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local brx = int.method.spatial_quadrature.weights
    local brt = int.method.time_quadrature.weights
    local Nbasis_λ_x = int.method.Nbasis_λ_x
    local Nbasis_μ_t = int.method.Nbasis_μ_t
    local mλ_x = int.method.mλ_x
    local mμ_t = int.method.mμ_t
    local show_status = int.method.show_status

    for d in 1:D
        for p in 1:NP
            z = zero(ST)
            for rt in 1:RT
                for rx in 1:RX
                    z += quad_b[rt, rx] *
                         (  x_domain * timestep(int) * C.∂L∂U_quad_values[d, rt, rx] * C_float.u_basis_quad_values[d,p,rt, rx]
                          + x_domain *                 C.∂L∂V_quad_values[d, rt, rx] * C_float.v_basis_quad_values[d,p,rt, rx]
                          + x_domain * timestep(int) * C.∂L∂W_quad_values[d, rt, rx] * C_float.w_basis_quad_values[d,p,rt, rx])
                end
            end
            for rx in 1:RX
                z += x_domain * brx[rx] * (C.λ₀_quad_values[d, rx] * C_float.ut₀_basis_quad_values[d,rx, p] - C.λ₁_quad_values[d, rx] * C_float.ut₁_basis_quad_values[d, rx, p])
            end
            for rt in 1:RT
                z += timestep(int) * brt[rt] * (C.μ₀_quad_values[d, rt] * C_float.ux₀_basis_quad_values[d, rt, p] - C.μ₁_quad_values[d, rt] * C_float.ux₁_basis_quad_values[d, rt, p])
            end
            b[p] = -z # TODO: check the sign
        end
    end

    for d in 1:D
        for p in 1:Nbasis_λ_x
            z = zero(ST)
            for rx in 1:RX
                z += x_domain * brx[rx] * mλ_x[p, rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
            end
            b[NP+(d-1)*Nbasis_λ_x+p] = -z
        end
    end

    for d in 1:D
        for p in 1:Nbasis_μ_t
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) * brt[rt] * mμ_t[p, rt] * (C.ux₀_quad_values[d, rt] - C.boundary_condition_x₀[d, rt])
            end
            b[NP+D*Nbasis_λ_x+(d-1)*Nbasis_μ_t+p] = -z
        end
    end

    for d in 1:D
        for p in 1:Nbasis_μ_t
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) * brt[rt] * mμ_t[p, rt] * (C.boundary_condition_x₁[d, rt] - C.ux₁_quad_values[d, rt])
            end
            b[NP+D*Nbasis_λ_x+D*Nbasis_μ_t+(d-1)*Nbasis_μ_t+p] = -z
        end
    end
    # @infiltrate
    if show_status
        @show b
    end
end

function update!(sol, int::PDEIntegrator{<:ELM_PDE_int})
    local D = int.problem.D
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local x = nlsolution(int)
    local h = timestep(int)
    local C = cache(int)
    local NP = int.method.basis.NP
    local v_basis_func = int.method.basis.v_basis
    local w_basis_func = int.method.basis.w_basis
    local u_basis_func = int.method.basis.u_basis
    local x_domain = xspan[2] - xspan[1]
    local nn_params = int.method.basis.u_basis.params
    local show_status = int.method.show_status
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local RX = int.method.RX
    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w


    x_nodes = collect(xspan[1]:xstep:xspan[2])
    ut₁_basis_values_tem = zeros(D, length(x_nodes), NP)
    vt₁_basis_values_tem = zeros(D, length(x_nodes), NP)
    wt₁_basis_values_tem = zeros(D, length(x_nodes), NP)


    for d in 1:D
        for i in eachindex(x_nodes)
            ut₁_basis_values_tem[d,i, :] = u_basis_func([1.0, x_nodes[i]], nn_params)
            vt₁_basis_values_tem[d,i, :] = v_basis_func([1.0, x_nodes[i]], nn_params)
            wt₁_basis_values_tem[d,i, :] = w_basis_func([1.0, x_nodes[i]], nn_params)
        end
    end


    for d in 1:D
        for i in eachindex(x_nodes)
            sol.u[i] = sum(ut₁_basis_values_tem[d,i, :] .* x[1:NP])
            sol.v[i] = sum(vt₁_basis_values_tem[d,i, :] .* x[1:NP]) / h
            sol.w[i] = sum(wt₁_basis_values_tem[d,i, :] .* x[1:NP])
        end
    end

    if show_status
         # boundary values at quadrature points
        ut₁_quad_values_truth = similar(C.ut₁_quad_values)
        vt₁_quad_values_truth = similar(C.vt₁_quad_values)
        wt₁_quad_values_truth = similar(C.wt₁_quad_values)

        for d in 1:D
            for j in 1:RX
                ut₁_quad_values_truth[d,j] = exact_u.(sol.t, x_quad_nodes[j])
                vt₁_quad_values_truth[d,j] = exact_v.(sol.t, x_quad_nodes[j])
                wt₁_quad_values_truth[d,j] = exact_w.(sol.t, x_quad_nodes[j])
            end

            for rx in 1:RX
                C.ut₁_quad_values[d,rx] = sum(C.ut₁_basis_quad_values[d,rx, :] .* x[1:NP])
                C.vt₁_quad_values[d,rx] = sum(C.vt₁_basis_quad_values[d,rx, :] .* x[1:NP]) / h
                C.wt₁_quad_values[d,rx] = sum(C.wt₁_basis_quad_values[d,rx, :] .* x[1:NP])
            end
        end

        @show maximum(abs.(C.ut₁_quad_values .- ut₁_quad_values_truth))
        @show maximum(abs.(C.vt₁_quad_values .- vt₁_quad_values_truth))
        @show maximum(abs.(C.wt₁_quad_values .- wt₁_quad_values_truth))

        ut₁_grid_truth = zeros(length(x_nodes))
        vt₁_grid_truth = zeros(length(x_nodes))
        wt₁_grid_truth = zeros(length(x_nodes))

        for i in eachindex(x_nodes)
            ut₁_grid_truth[i] = exact_u.(sol.t, x_nodes[i])
            vt₁_grid_truth[i] = exact_v.(sol.t, x_nodes[i])
            wt₁_grid_truth[i] = exact_w.(sol.t, x_nodes[i])
        end

        @show maximum(abs.(sol.u .- ut₁_grid_truth))
        @show maximum(abs.(sol.v .- vt₁_grid_truth))
        @show maximum(abs.(sol.w .- wt₁_grid_truth))

        @show sol.u
        @show ut₁_grid_truth
    end
end
