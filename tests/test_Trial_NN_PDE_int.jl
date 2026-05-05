# cd("MultiSymplectic.jl")
# using Pkg
# Pkg.activate(".")
using Revise
using GeometricIntegratorsBase
using MultiSymplectic
using AbstractNeuralNetworks
using Zygote
using Plots
using Profile
using Base
using Infiltrator
# t_step = parse(Float64, ARGS[1])
# rt = parse(Int, ARGS[2])
# Nw = parse(Int, ARGS[3])
# Nb = parse(Int, ARGS[4])

t_step = 0.5
NN_width = 70

GeometricIntegratorsBase.default_options(::TrialNN_PDE_int) = (
    max_iterations = 100,
    regularization_factor = 1e-5,
    f_abstol = 2eps(),
    x_suctol = 2eps()
)

x_step = 0.01
x_span = (0.0, 1.0)
lpde = MultiSymplectic.Wave.lpdeproblem(timestep=t_step, timespan=(0.0, t_step), xspan=x_span, xstep=x_step)

relu3(x) = max(0.0, x)^3
activation = tanh
trial_basis = Trial_Solution_Basis(NN_width, activation,lpde.exact_u,x_span)

# for rt in [8,12,24]
# for rx in [16, 32, 64]
# log_file = "logs/trial_nn_int.txt"
# open(log_file, "w") do io
#         redirect_stdio(stdout=log_file, stderr=log_file) do
                trial_int = TrialNN_PDE_int(trial_basis, show_status=false)
                sol= MultiSymplectic.integrate(lpde,trial_int)
        
                p = @layout [a b c]
                p1 = plot([lpde.exact_u(t_step,xx) for xx in x_span[1]:0.01:x_span[2]], label="exact_u")
                plot!(p1, sol.sol.s.u[1], label="sol.u")
                p2 = plot([lpde.exact_v(t_step,xx) for xx in x_span[1]:0.01:x_span[2]], label="exact_v")
                plot!(p2, sol.sol.s.v[1], label="sol.v")
                p3 = plot([lpde.exact_w(t_step,xx) for xx in x_span[1]:0.01:x_span[2]], label="exact_w")
                plot!(p3, sol.sol.s.w[1], label="sol.w")
                p = plot(p1, p2, p3, layout=p)
                savefig("logs/trial_nn_int_t=$t_step.pdf")

#         end
# end
# end
# end
# (u_trial(grid_matrix[1,1][1] , xspan[1] + x_domain* grid_matrix[1,1][2]+ 3eps(),x,W1,bias1 ) - u_trial(grid_matrix[1,1][1] , xspan[1] + x_domain* grid_matrix[1,1][2]- 3eps(),x,W1,bias1 )) / 6eps()



# # prepare x values for plotting
# x_plot = collect(x_span[1]:x_step:x_span[2])

# # First subplot: NN vs exact at t = h (t_step)
# p1 = plot(trial_sol.sol.u[1], label="NN Solution", title="t = $t_step", xlabel="x", ylabel="u")
# plot!(p1, lpde.exact_u.(t_step, x_plot), label="Exact")

# # # Second subplot: NN vs exact at t = 2h (2*t_step)
# p2 = plot(trial_sol.sol.u[2], label="NN Solution", title="t = $(2*t_step)", xlabel="x", ylabel="u")
# plot!(p2, lpde.exact_u.(2*t_step, x_plot), label="Exact")

# # Combine into a 1x2 layout and save
# plot(p1, p2, layout=(1,2), size=(1000,400))
# savefig("logs/h$(t_step)_NNwidth$(NN_width)_Nw$(Nw)_Nb$(Nb)_RT$(rt)_RX$(rx)$(activation)_full.png")
#         #     end
#         # end
# #     end
# # end



# wave_anim = @animate for (i, tt) in enumerate(0:0.1:2)
#     plot(lpde.exact_u.(tt, collect(-1:0.01:1)), label="Exact")

#     title!("t = $tt")
#     xlabel!("x")
#     ylabel!("u")
# end
# gif(wave_anim, fps=5)

# a = 0
# b = 1
# x_domain = b - a
# h = 0.3

# exact_u(t, x) = lpde.exact_u(t, x)

# # #  Trial solution function construction
# # psi_L(x) = (b - x) / x_domain       # left 
# # psi_R(x) = (x - a) / x_domain       # right
# # phi_B(t) = (h - t) / h   # bottom

# # function T1NN(t, x, dofs)
# #     return (b - x) * dofs' * PNN([t, a]) / x_domain +
# #            (x - a) * dofs' * PNN([t, b]) / x_domain +
# #            (h - h * t) * dofs' * PNN([0.0, x]) / h
# # end

# # function T1NN(t, x)
# #     return (b - x) * W2 * tanh(W1[1] * t + W1[2] * a + bias) / x_domain +
# #            (x - a) * W2 * tanh(W1[1] * t + W1[2] * b + bias) / x_domain +
# #            (h - h * t) * W2 * tanh(W1[2] * x + bias) / h
# # end

# # function T2NN(t, x, dofs)
# #     return (b - x) * (h - h * t) * dofs' * PNN([0.0, a]) / x_domain / h +
# #            (x - a) * (h - h * t) * dofs' * PNN([0.0, b]) / x_domain / h
# # end

# # function C1(t, x)
# #     return (b - x) * exact_u(t, a) / x_domain +
# #            (x - a) * exact_u(t, b) / x_domain +
# #            (h - h * t) * exact_u(0.0, x) / h
# # end


# # function C2(t, x)
# #     return (b - x) * (h - h * t) * exact_u(0, a) / x_domain / h +
# #            (x - a) * (h - h * t) * exact_u(0, b) / x_domain / h
# # end

# # u_trial(t, x, dofs) = dofs' * PNN([t, x]) - T1NN(t, x, dofs) + T2NN(t, x, dofs) + C1(t, x) - C2(t, x)

# # x_ls = collect(a:0.01:b)
# # t_ls = collect(0.0:0.01:h)
# # u0 = rand(NN_width)
# # u_vals = [u_trial(ti,xi,u0) for ti in t_ls, xi in x_ls]
# # truth_vals = [exact_u(ti,xi) for ti in t_ls, xi in x_ls]
# # u_vals - truth_vals


# # v_trial(t,x,dofs) = Zygote.gradient(tt -> u_trial(tt,x,dofs),t)[1]
# # w_trial(t,x,dofs) = Zygote.gradient(xx -> u_trial(t,xx,dofs),x)[1]

# N_in = 1000
# tx_in = rand(2, N_in)
# tx_in[1, :] = tx_in[1, :]
# tx_in[2, :] = a .+ x_domain * tx_in[2, :]


# # utt(t,x,dofs) = ForwardDiff.derivative(tt -> ForwardDiff.derivative(ttt -> u_trial(ttt, x, dofs), tt), t)
# # uxx(t,x,dofs) = ForwardDiff.derivative(xx -> ForwardDiff.derivative(xxx -> u_trial(t, xxx, dofs), xx), x)

# function nlls!(du, u, p)
#     for i in 1:N_in
#         t, x = tx_in[1, i], tx_in[2, i]
#         # Second derivatives
#         # Wave equation: ∂²u/∂t² - c²∂²u/∂x² = 0
#         # du[i] = utt(t,x,u) -  uxx(t,x,u)
#         du[i] = u_trial(t, x, u) - exact_u(t, x)
#     end
# end

# using NonlinearSolve
# u0 = zeros(NN_width)
# prob = NonlinearLeastSquaresProblem(
#     NonlinearFunction(nlls!, resid_prototype=zeros(N_in)), u0)

# @time sol = solve(prob, LevenbergMarquardt(), maxiters=100, abstol=1e-6, reltol=1e-6)
# u_sol = sol.u

# t_plot = 0:0.01:h
# x_plot = 0:0.01:1
# u_pred = [u_trial(t, x, sol.u) for t in t_plot, x in x_plot]
# u_exact = [exact_u(t, x) for t in t_plot, x in x_plot]
# error_plot = abs.(u_pred - u_exact)

# # Create plots
# p1 = surface(t_plot, x_plot, u_pred', title="Neural Network Solution", xlabel="t", ylabel="x_domain", zlabel="u")
# p2 = surface(t_plot, x_plot, u_exact', title="Exact Solution", xlabel="x", ylabel="y", zlabel="u")
# p3 = surface(t_plot, x_plot, error_plot', title="Absolute Error", xlabel="x", ylabel="y", zlabel="Error")

# plot(p1, p2, p3, layout=(1, 3), size=(1500, 400))


# begin
#     ########
#     using AbstractNeuralNetworks
#     a = 0
#     b = 1
#     x_domain = b - a
#     h = 0.3
#     c = 0.5
#     A1 = 0.05
#     B1 = 0.0
#     l = 1.0

#     target_function(t, x) = (A1 * cos((pi * c * t) / l) + B1 * sin((pi * c * t) / l + pi / 6)) * sin((pi * x) / l)
#     function T1NN_BNN(t, x,BNN; ps = BNN.params)
#         return (b - x) / x_domain * BNN([t, a],ps) +
#                (x - a) / x_domain * BNN([t, b],ps) +
#                (1 - t) * BNN([0.0, x],ps)
#     end

#     function T1NN_BNN(t::Vector{Float64}, x::Vector{Float64},BNN; ps = BNN.params)
#         results = zeros(NN_width, N_train)
#         for i in 1:length(t)
#             results[:, i] = T1NN_BNN(t[i], x[i],BNN; ps)
#         end
#         return results
#     end

#     function T2NN_BNN(t, x,BNN; ps = BNN.params)
#         return (b - x) / x_domain * (1 - t) * BNN([0.0, a],ps) +
#                (x - a) / x_domain * (1 - t) * BNN([0.0, b],ps)
#     end

#     function T2NN_BNN(t::Vector{Float64}, x::Vector{Float64},BNN; ps = BNN.params)
#         results = zeros(NN_width, N_train)
#         for i in 1:length(t)
#             results[:, i] = T2NN_BNN(t[i], x[i],BNN; ps)
#         end
#         return results
#     end

#     function init_function(t, x)
#         target_function(0, x)
#     end

#     function C1(t, x)
#         return (b - x) / x_domain * target_function(t, a) +
#                (x - a) / x_domain * target_function(t, b) +
#                (1 - t) * target_function(0, x)
#     end

#     function C2(t, x)
#         return (b - x) / x_domain * (1 - t) * target_function(0, a) +
#                (x - a) / x_domain * (1 - t) * target_function(0, b)
#     end

#     relu2(x) = max(0, x)^3
#     beta = 0.5
#     swish(x) = x / (1 + exp(-beta * x))
#     NN_width = 100
#     BNN = NeuralNetwork(Chain(
#         Dense(2, NN_width, relu2),
#     ))

#     N_train = 800
#     collocation_points = rand(2, N_train)
#     collocation_points[2, :] = a .+ (b - a) * collocation_points[2, :]

#     rhs = target_function.(h .* collocation_points[1, :], collocation_points[2, :]) .- C1.(collocation_points[1, :], collocation_points[2, :]) .+ C2.(collocation_points[1, :], collocation_points[2, :])
#     A = BNN(collocation_points) .- T1NN_BNN(collocation_points[1, :], collocation_points[2, :],BNN) .+ T2NN_BNN(collocation_points[1, :], collocation_points[2, :],BNN)
#     sol = A' \ rhs

#     prob = LinearProblem(A', rhs)
#     sol2 = LinearSolve.solve(prob, KrylovJL_LSMR())


#     t_plot = 0:0.01:h
#     x_plot = a:0.01:b
#     u_pred = [sol2' * (BNN([t, x]) - T1NN_BNN(t, x,BNN) + T2NN_BNN(t, x,BNN)) + C1(t, x) - C2(t, x) for t in t_plot, x in x_plot]
#     u_exact = [target_function(t, x) for t in t_plot, x in x_plot]
#     error_plot = abs.(u_pred - u_exact)
#     @show maximum(error_plot)


#     using Plots
#     # Create plots
#     p1 = surface(t_plot, x_plot, u_pred', title="Neural Network Solution", xlabel="t", ylabel="x", zlabel="u")
#     p2 = surface(t_plot, x_plot, u_exact', title="Exact Solution", xlabel="t", ylabel="x", zlabel="u")
#     p3 = surface(t_plot, x_plot, error_plot', title="Absolute Error", xlabel="t", ylabel="x", zlabel="Error")

#     plot(p1, p2, p3, layout=(1, 3), size=(1500, 400))
# end


# # initialize the parameters and train with LSGD
# a = 0
# b = 1
# x_domain = b - a
# h = 0.3

# c = 0.5
# A1 = 0.05
# B1 = 0.0
# l = 1.0
# target_function(t, x) = (A1 * cos((pi * c * t) / l) + B1 * sin((pi * c * t) / l + pi / 6)) * sin((pi * x) / l)
# # target_function(t, x) = -exp(cos(pi * x + 3*pi) + t^2)

# using GeometricMachineLearning
# using Statistics
# using LinearSolve
# using Zygote
# using Plots
# using Random


# function C1(t, x)
#     return (b - x) / x_domain * target_function(t, a) +
#             (x - a) / x_domain * target_function(t, b) +
#             (1 - t) * target_function(0, x)
# end

# function C2(t, x)
#     return (b - x) / x_domain * (1 - t) * target_function(0, a) +
#             (x - a) / x_domain * (1 - t) * target_function(0, b)
# end

# function lsgd_loss(network_inputs, labels,  ps)
#     NN_output = [u_trial(network_inputs[1,i],network_inputs[2,i], ps) for i in eachindex(network_inputs[1,:])]
#     return sqrt(Statistics.mean((labels .- NN_output) .^ 2))
# end



# function T1NN(t, x,params)
#     return (b - x) / x_domain * PNN([t, a], params)[1] +
#            (x - a) / x_domain * PNN([t, b], params)[1] +
#            (h - h * t) / h * PNN([0.0, x],params)[1]
# end

# function T2NN(t, x, params)
#     return (b - x) / x_domain * (h - h * t) / h * PNN([0.0, a], params)[1] +
#            (x - a) / x_domain * (h - h * t) / h * PNN([0.0, b], params)[1]
# end

# function C1(t, x)
#     return (b - x) * target_function(t, a) / x_domain +
#            (x - a) * target_function(t, b) / x_domain +
#            (h - h * t) * target_function(0.0, x) / h
# end


# function C2(t, x)
#     return (b - x) * (h - h * t) * target_function(0, a) / x_domain / h +
#            (x - a) * (h - h * t) * target_function(0, b) / x_domain / h
# end

# function u_trial(t, x, params)
#     PNN([t, x], params)[1] - T1NN(t, x,params) + T2NN(t, x,params) + C1(t, x) - C2(t, x)
# end


# t_plot = 0:0.01:h
# x_plot = a:0.01:b



# for S in [320,]#50,100,200,300
#     for N_train = [320,]#, 800,1000,1500,2000
#         for lambda = [0.1,0.01,0.001]

#         function T1NN_BNN(t, x,BNN; ps = BNN.params)
#             return (b - x) / x_domain * BNN([t, a],ps) +
#                     (x - a) / x_domain * BNN([t, b],ps) +
#                     (1 - t) * BNN([0.0, x],ps)
#         end

#         function T1NN_BNN(t::Vector{Float64}, x::Vector{Float64},BNN; ps = BNN.params)
#             results = zeros(S, N_train)
#             for i in 1:length(t)
#                 results[:, i] = T1NN_BNN(t[i], x[i],BNN; ps)
#             end
#             return results
#         end

#         function T2NN_BNN(t, x,BNN; ps = BNN.params)
#             return (b - x) / x_domain * (1 - t) * BNN([0.0, a],ps) +
#                     (x - a) / x_domain * (1 - t) * BNN([0.0, b],ps)
#         end

#         function T2NN_BNN(t::Vector{Float64}, x::Vector{Float64},BNN; ps = BNN.params)
#             results = zeros(S, N_train)
#             for i in 1:length(t)
#                 results[:, i] = T2NN_BNN(t[i], x[i],BNN; ps)
#             end
#             return results
#         end


#             # N_train = 1000
#             collocation_points = rand(Random.default_rng(1),2, N_train)
#             collocation_points[2, :] = a .+ (b - a) * collocation_points[2, :]
#             collocation_points[1, :] = h .* collocation_points[1, :]

#             # NN_width = S
#             # sol = rand(S)
#             # relu2(x) = max(0, x)^2
#             BNN = NeuralNetwork(Chain(Dense(2, S, tanh),))
#             # PNN = NeuralNetwork(Chain(Dense(2, S, tanh), Dense(S, 1, identity, use_bias=false)))


#             rhs = target_function.(h .* collocation_points[1, :], collocation_points[2, :]) .- C1.(collocation_points[1, :], collocation_points[2, :]) .+ C2.(collocation_points[1, :], collocation_points[2, :])
#             A = BNN(collocation_points,BNN.params) .- T1NN_BNN(collocation_points[1, :], collocation_points[2, :],BNN, ps = BNN.params) .+ T2NN_BNN(collocation_points[1, :], collocation_points[2, :], BNN, ps = BNN.params)

#             # prob = LinearProblem(A'A + lambda .* Matrix(I, S, S), A' * rhs)
#             prob = LinearProblem(A', rhs)
#             sol = LinearSolve.solve(prob, KrylovJL_LSMR())

#             # u_pred = [sol' * (BNN([t, x]) - T1NN_BNN(t, x,BNN, ps = BNN.params) + T2NN_BNN(t, x,BNN, ps = BNN.params)) + C1(t, x) - C2(t, x) for (t,x) in zip(collocation_points[1, :], collocation_points[2, :])]


#             u_pred = [sol' * (BNN([t, x]) - T1NN_BNN(t, x,BNN, ps = BNN.params) + T2NN_BNN(t, x,BNN, ps = BNN.params)) + C1(t, x) - C2(t, x) for t in t_plot, x in x_plot]
#             u_exact = [target_function(t, x) for t in t_plot, x in x_plot]
#             error_plot = abs.(u_pred - u_exact)
#             surface(t_plot, x_plot, error_plot', title="Absolute Error", xlabel="t", ylabel="x", zlabel="Error")

#             @show maximum(error_plot)
#         end
#     end
# end
# surface(t_plot, x_plot, error_plot', title="Absolute Error", xlabel="t", ylabel="x", zlabel="Error")
# surface(t_plot, x_plot, u_exact', xlabel="t", ylabel="x", zlabel="Error")

# # error_plot = abs.(u_pred - u_exact)
# println(" max error : ",maximum(error_plot))

# for (name, layer) in zip(keys(PNN.params), values(PNN.params))
#     in_size = size(layer.W, 2)
#     out_size = size(layer.W, 1)
#     if hasfield(typeof(layer), :b)
#         PNN.params[name].W[:], PNN.params[name].b[:] = BNN.params[name].W[:], BNN.params[name].b[:]
#     else
#         # For layers without bias (e.g., output), just regenerate W
#         PNN.params[name].W[:] = sol
#     end
# end

# tem_ps = PNN.params[keys(PNN.params)[1:end-1]]
# opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.GradientOptimizer(0.005), tem_ps)
# # opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.AdamOptimizer(0.01), tem_ps)

# err = 0
# λ = GeometricMachineLearning.GlobalSection(tem_ps)


# nepochs = 200




# u_exact = [target_function(t, x) for (t,x ) in  zip(collocation_points[1, :], collocation_points[2, :])]

# ds_loss = zeros(nepochs)
# ls_loss = zeros(nepochs)

# # ps = PNN.params 
# NN_output = [u_trial(collocation_points[1,i],collocation_points[2,i], PNN.params) for i in eachindex(collocation_points[2,:])]

# u_pred = [sol' * (BNN([t, x],BNN.params)- T1NN_BNN(t, x,BNN, ps = BNN.params) + T2NN_BNN(t, x,BNN, ps = BNN.params)) + C1(t, x) - C2(t, x) for (t,x) in zip(collocation_points[1,:], collocation_points[2,:])]





# for ep in 1:nepochs
#     # @show BNN.params.L1.W
#     # @show BNN.params.L1.b

#     # @show PNN.params.L1.W
#     # @show PNN.params.L1.b
#     # @show PNN.params.L2.W
#     rhs = target_function.(h .* collocation_points[1, :], collocation_points[2, :]) 
#     A = BNN(collocation_points,BNN.params) .- T1NN_BNN(collocation_points[1, :], collocation_points[2, :],BNN, ps = BNN.params) .+ T2NN_BNN(collocation_points[1, :], collocation_points[2, :], BNN, ps = BNN.params) .+ C1.(collocation_points[1, :], collocation_points[2, :]) .- C2.(collocation_points[1, :], collocation_points[2, :])
#     # Φ = NN(network_inputs, PNN.params)
#     # PNN.params.L3.W[:] = labels/Φ

#     prob = LinearProblem(A', rhs)
#     sol = LinearSolve.solve(prob, KrylovJL_LSMR())
#     # @show sol

#     PNN.params.L2.W[:] = sol
#     PNN.params.L1.W[:] = BNN.params.L1.W[:]
#     PNN.params.L1.b[:] = BNN.params.L1.b[:]

#     u_pred = [sol' * (BNN([t, x]) - T1NN_BNN(t, x,BNN, ps = BNN.params) + T2NN_BNN(t, x,BNN, ps = BNN.params)) + C1(t, x) - C2(t, x) for (t,x) in zip(collocation_points[1, :], collocation_points[2, :])]
#     error_plot = abs.(u_pred - u_exact)

#     # println("After least square, Epoch: " ,ep, " max error : ",maximum(error_plot))
#     # ls_loss[ep] = maximum(error_plot)
#     err = lsgd_loss(collocation_points, rhs, PNN.params)
#     println("After lsq, lsgd loss:",err)
#     ls_loss[ep] = err



#     for (name, layer) in zip(keys(PNN.params), values(PNN.params))
#         in_size = size(layer.W, 2)
#         out_size = size(layer.W, 1)
#         if hasfield(typeof(layer), :b)
#             PNN.params[name].W[:] = BNN.params[name].W[:]
#             PNN.params[name].b[:] = BNN.params[name].b[:]
#         else
#             # For layers without bias (e.g., output), just regenerate W
#             PNN.params[name].W[:] = sol
#         end
#     end

#     err = lsgd_loss(collocation_points, rhs, PNN.params)
#     println("After copy params, lsgd loss:",err)

#     # @show PNN.params.L1.W
#     # @show PNN.params.L1.b
#     # @show PNN.params.L2.W


#     gs = Zygote.gradient(p -> lsgd_loss(collocation_points, rhs, p), PNN.params)[1]
#     # @show gs
#     # tem_ps = PNN.params[keys(PNN.params)[1:end-1]]
#     # tem_gs = gs[keys(gs)[1:end-1]]
#     # @show tem_ps
#     # GeometricMachineLearning.optimization_step!(opt, λ, tem_ps, tem_gs)

#     PNN.params.L1.W[:] .= PNN.params.L1.W[:] .- 0.1 * gs.L1.W[:]
#     PNN.params.L1.b[:] .= PNN.params.L1.b[:] .- 0.1 * gs.L1.b[:]

#     err = lsgd_loss(collocation_points, rhs, PNN.params)
#     println("After gradient descent, lsgd loss:",err)

#     # @show PNN.params.L1.W
#     # @show PNN.params.L1.b
#     # @show PNN.params.L2.W


#     # sol = PNN.params.L2.W[:]
#     # BNN.params.L1.W[:] = PNN.params.L1.W[:]
#     # BNN.params.L1.b[:] = PNN.params.L1.b[:]

#     for (name, layer) in zip(keys(PNN.params), values(PNN.params))
#         in_size = size(layer.W, 2)
#         out_size = size(layer.W, 1)
#         if hasfield(typeof(layer), :b)
#             BNN.params[name].W[:] = PNN.params[name].W[:]
#             BNN.params[name].b[:] = PNN.params[name].b[:]
#         else
#             # For layers without bias (e.g., output), just regenerate W
#             sol = PNN.params[name].W[:]
#         end
#     end

#     u_pred = [sol' * (BNN([t, x],BNN.params) - T1NN_BNN(t, x,BNN, ps = BNN.params) + T2NN_BNN(t, x,BNN, ps = BNN.params)) + C1(t, x) - C2(t, x) for (t,x) in zip(collocation_points[1, :], collocation_points[2, :])]
#     # error_plot = abs.(u_pred - u_exact)
#     # println("After Descent, Epoch: " ,ep, "error : ",maximum(error_plot))

#     ds_loss[ep] = err
#     # println("epochs:",ep, "loss,",err)
#     if err < 5e-8
#         print("\n final loss: $err by $ep epochs")
#         break
#     elseif ep == nepochs
#         print("\n final loss: $err by $ep epochs")
#     end
# end 

# plot(ds_loss)
# plot(ls_loss)


# using Symbolics, LinearAlgebra
# using Symbolics: IfElse

# # 假设 S 是隐藏层神经元数量
# function generate_symbolic_u_trial(S::Int, activation_fn, exact_u_sym_fn; a=0.0, b=1.0, h=0.1, tn=0.0)
#     # 1. 定义符号变量
#     @variables t x
#     @variables W2[1:S] W1[1:S, 1:2] bias1[1:S]
#     # 如果 C1 中涉及上一时刻的权重，也需要定义为符号常数（或者直接传入数值）
#     @variables pW2[1:S] pW1[1:S, 1:2] pbias1[1:S]

#     # 2. 定义符号 NN
#     # 映射激活函数 (如果是自定义函数，需要先注册，例如 @register_symbolic my_act(x))
#     # 假设 activation_fn 是可以直接处理 Symbolics.Num 的函数，如 σ(x) = 1/(1+exp(-x))
#     function sym_NN(_t, _x, _W2, _W1, _b1)
#         return sum(_W2[i] * activation_fn(_W1[i,1]*_t + _W1[i,2]*_x + _b1[i]) for i in 1:S)
#     end

#     # 3. 构造辅助项 (T1, T2, C1, C2) 的符号表达式
#     x_domain = b - a
    
#     # T1NN_manual
#     t1_nn = (b - x) / x_domain * sym_NN(t, a, W2, W1, bias1) +
#             (x - a) / x_domain * sym_NN(t, b, W2, W1, bias1) +
#             (h - h * t) / h * sym_NN(0.0, x, W2, W1, bias1)

#     # T2NN_manual
#     t2_nn = (b - x) / x_domain * (h - h * t) / h * sym_NN(0.0, a, W2, W1, bias1) +
#             (x - a) / x_domain * (h - h * t) / h * sym_NN(0.0, b, W2, W1, bias1)

#     # C1 和 C2
#     # 注意：exact_u 必须能接受符号输入。如果是闭式解，直接写表达式。
#     # 如果 exact_u 是黑盒，需要用 @register_symbolic 注册
#     if tn == 0.0
#         c1 = (b - x) * exact_u_sym_fn(h*t, a) / x_domain +
#              (x - a) * exact_u_sym_fn(h*t, b) / x_domain +
#              (h - h * t) * exact_u_sym_fn(tn, x) / h
#     else
#         # 对应原有逻辑中 tn != 0 使用上一时刻 NN 的部分
#         c1 = (b - x) * exact_u_sym_fn(tn + h*t, a) / x_domain +
#              (x - a) * exact_u_sym_fn(tn + h*t, b) / x_domain +
#              (h - h * t) * sym_NN(1.0, x, pW2, pW1, pbias1) / h
#     end

#     c2 = (b - x) * (h - h * t) * exact_u_sym_fn(tn, a) / x_domain / h +
#          (x - a) * (h - h * t) * exact_u_sym_fn(tn, b) / x_domain / h

#     # 4. 组合得到 u_trial 符号表达式
#     expr_u = sym_NN(t, x, W2, W1, bias1) - t1_nn + t2_nn + c1 - c2

#     return expr_u, (t, x), (W2, W1, bias1, pW2, pW1, pbias1)
# end

# # 假设配置
# S = 50
# σ(x) = tanh(x) # 激活函数
# exact_u_expr(t, x) = sin(pi*x) * exp(-t) # 举例一个精确解

# # 生成表达式
# u_expr, (t_sym, x_sym), (W2_s, W1_s, b1_s, pW2_s, pW1_s, pb1_s) = 
#     generate_symbolic_u_trial(S, σ, exact_u_expr,tn=2.0)

# # 1. 对时间 t 求导 (v_trial)
# v_expr = Symbolics.derivative(u_expr, t_sym)

# # 2. 对空间 x 求导 (w_trial)
# w_expr = Symbolics.derivative(u_expr, x_sym)

# # 3. 对参数 p 求导 (∂u∂p)
# # 把所有训练参数打包成一个向量
# params_flat = [vec(W2_s); vec(W1_s); vec(b1_s)]
# du_dp_expr = Symbolics.jacobian([u_expr], params_flat) 
# dv_dp_expr = Symbolics.jacobian([v_expr], params_flat)
# dw_dp_expr = Symbolics.jacobian([w_expr], params_flat)

# # 4. 编译为 Julia 函数
# # build_function 会生成非常高效的、包含展开循环的代码
# # target=:function 生成普通函数，target=:inplace 可以生成不分配内存的版本
# u_func = build_function(u_expr, t_sym, x_sym, params_flat, pW2_s, pW1_s, pb1_s, expression=Val{false})
# v_func = build_function(v_expr, t_sym, x_sym, params_flat, pW2_s, pW1_s, pb1_s, expression=Val{false})
# w_func = build_function(w_expr, t_sym, x_sym, params_flat, pW2_s, pW1_s, pb1_s, expression=Val{false})

# ∂u∂p_func = build_function(du_dp_expr, t_sym, x_sym, params_flat, pW2_s, pW1_s, pb1_s, expression=Val{false})[1]
# ∂v∂p_func = build_function(dv_dp_expr, t_sym, x_sym, params_flat, pW2_s, pW1_s, pb1_s, expression=Val{false})[1]
# ∂w∂p_func = build_function(dw_dp_expr, t_sym, x_sym, params_flat, pW2_s, pW1_s, pb1_s, expression=Val{false})[1]

# t_val = 0.5
# x_val = 0.2
# tn_val = 2.0
# h_val = 0.1
# xspan = x_span
# p_current = rand(4*S) # W2, W1, bias 的平坦化向量
# pw2 = rand(S)
# pw1 = rand(S, 2)
# pb1 = rand(S)

# # 直接调用
# @benchmark ∂v∂p_func(t_val, x_val,tn_val, h_val, p_current, pw2, pw1, pb1)
# @benchmark ∂v∂p(t_val, x_val, p_current)




# c=0.5
# A1 = 0.8
# A2 = 0.0
# B1 = 0.8
# B2 = 0.0
# l = 1.0
# activation = tanh
# a,b = 0.0, 1.0
# x_domain = b - a
# h = h_val
# tn = tn_val
# function exact_u(t,x)
# (A1 * cos((pi*c*t)/l) + B1 * sin((pi*c*t)/l + pi/6)) * sin((pi*x)/l) + (A2 * cos((2*pi*c*t)/l) + B2 * sin((2*pi*c*t)/l + pi/6)) * sin((2*pi*x)/l)
# end

# function NN(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}) where {TT,XT, DT}
#     return sum(W2[i] * activation(W1[i,1]*t + W1[i,2]*x + bias1[i]) for i in eachindex(W2))
# end

# function T1NN_manual(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}) where {TT,XT, DT}

#     return (b - x) / x_domain * NN(t, a, W2,W1,bias1) +
#            (x - a) / x_domain * NN(t, b, W2,W1,bias1) +
#            (h - h * t) / h * NN(0.0, x, W2,W1,bias1)
# end

# function T2NN_manual(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}) where {TT,XT, DT}

#     return (b - x) / x_domain * (h - h * t) / h * NN(0.0, a, W2,W1,bias1) +
#            (x - a) / x_domain * (h - h * t) / h * NN(0.0, b, W2,W1,bias1)
# end

# function C1(t::TT, x::XT, tn::ST) where {TT,XT,ST}
#     local a,b = xspan[1],xspan[2]
#     local x_domain = b-a

#     return (b - x) * exact_u(h*t, a) / x_domain +
#         (x - a) * exact_u(h*t, b) / x_domain +
#         (h - h * t) * exact_u(tn, x) / h

# end

# function C2(t::TT, x::XT, tn::ST) where {TT,XT,ST}

#     return (b - x) * (h - h * t) * exact_u(tn, a) / x_domain / h +
#            (x - a) * (h - h * t) * exact_u(tn, b) / x_domain / h
# end

# function u_trial(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}) where {TT,XT, DT}
#     NN(t, x, W2, W1, bias1) - T1NN_manual(t, x, W2, W1, bias1) + T2NN_manual(t, x, W2, W1, bias1) + C1(t, x, tn) - C2(t, x, tn)
# end

# function u_trial(t::TT,x::XT,all_params::AbstractVector{DT} ) where {TT,XT, DT}
#     @views W2 = all_params[1:S]
#     @views W1 = reshape(all_params[S+1:3*S], S, 2)
#     @views bias1 = all_params[3*S+1:4*S]
#     return u_trial(t,x,W2,W1,bias1)
# end

# u_trial(tx::Vector{ST},all_params::Vector{DT}) where {ST, DT} = u_trial(tx[1],tx[2],all_params) 
# u_trial(tx::NTuple{2,ST},all_params::Vector{DT} ) where {ST<:Real, DT} = u_trial(tx[1],tx[2],all_params)
# v_trial_zygote(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}, sol) where {TT,XT, DT} = Zygote.gradient(tt -> u_trial(tt,x,W2,W1,bias1 ),t)[1]
# w_trial_zygote(t::TT, x::XT, W2::AbstractVector{DT}, W1::AbstractMatrix{DT}, bias1::AbstractVector{DT}, sol) where {TT,XT, DT} = Zygote.gradient(xx -> u_trial(t,xx,W2,W1,bias1 ),x)[1]


# u_trial(tx::Vector{ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, sol) where {ST, DT} = u_trial(tx[1],tx[2],W2,W1,bias1 )
# u_trial(tx::NTuple{2,ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, sol) where {ST<:Real, DT} = u_trial(tx[1],tx[2],W2,W1,bias1 )
# v_trial_zygote(tx::Vector{ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, sol) where {ST, DT} = v_trial_zygote(tx[1],tx[2],W2,W1,bias1 )
# v_trial_zygote(tx::NTuple{2,ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, sol) where {ST<:Real, DT} = v_trial_zygote(tx[1],tx[2],W2,W1,bias1 )
# w_trial_zygote(tx::Vector{ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, sol) where {ST, DT} = w_trial_zygote(tx[1],tx[2],W2,W1,bias1 )
# w_trial_zygote(tx::NTuple{2,ST}, W2::Vector{DT}, W1::Matrix{DT}, bias1::Vector{DT}, sol) where {ST<:Real, DT} = w_trial_zygote(tx[1],tx[2],W2,W1,bias1 )


# v_trial(t::TT,x::XT,params::AbstractVector{DT} ) where {TT,XT, DT} = ForwardDiff.derivative(tt -> u_trial(tt,x,params ),t)[1]
# w_trial(t::TT,x::XT,params::AbstractVector{DT} ) where {TT,XT, DT} = ForwardDiff.derivative(xx -> u_trial(t,xx,params ),x)[1]
    

# ∂u∂p(t::TT,x::XT,params::AbstractVector{DT} ) where {TT,XT, DT} = ForwardDiff.gradient(p -> u_trial(t,x,p ),params)
# ∂v∂p(t::TT,x::XT,params::AbstractVector{DT} ) where {TT,XT, DT} = ForwardDiff.gradient(p -> v_trial(t,x,p ),params)
# ∂w∂p(t::TT,x::XT,params::AbstractVector{DT} ) where {TT,XT, DT} = ForwardDiff.gradient(p -> w_trial(t,x,p ),params)

# ∂u∂p(input::Vector{ST}, params::Vector{DT}, sol) where {ST, DT} = ∂u∂p(input[1],input[2],params )
# ∂v∂p(input::Vector{ST}, params::Vector{DT}, sol) where {ST, DT} = ∂v∂p(input[1],input[2],params )
# ∂w∂p(input::Vector{ST}, params::Vector{DT}, sol) where {ST, DT} = ∂w∂p(input[1],input[2],params )
# ∂u∂p(input::NTuple{2,ST}, params::Vector{DT}, sol) where {ST<:Real, DT} = ∂u∂p(input[1],input[2],params )
# ∂v∂p(input::NTuple{2,ST}, params::Vector{DT}, sol) where {ST<:Real, DT} = ∂v∂p(input[1],input[2],params )
# ∂w∂p(input::NTuple{2,ST}, params::Vector{DT}, sol) where {ST<:Real, DT} = ∂w∂p(input[1],input[2],params )

