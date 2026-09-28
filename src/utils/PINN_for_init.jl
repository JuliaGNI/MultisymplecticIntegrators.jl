using AbstractNeuralNetworks
using GeometricMachineLearning
using MultisymplecticIntegrators
using LinearAlgebra
using Statistics
using Zygote
using IterTools
using Random
using ForwardDiff
using NonlinearSolve

u_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 1, identity, use_bias = false)
)

PNN = NeuralNetwork(u_network)
# exact_u = MultisymplecticIntegrators.LinearTransport.exact_u
# exact_u(0.0,0.5)

A1 = 0.4
B1 = 0.3
A2 = 0.0
B2 = 0.0
c = 1.0
function exact_u(t, x)
    (A1 * cos((pi*c*t)) + B1 * sin((pi*c*t) + pi/6)) * sin((pi*x)) +
    (A2 * cos((2*pi*c*t)) + B2 * sin((2*pi*c*t) + pi/6)) * sin((2*pi*x)) + 1.0
end

tstep = 0.3
tspan = (0.0, 1.5)
xspan = (0.0, 1.0)
a, b = xspan[1], xspan[2]
x_domain = xspan[2] - xspan[1]
h = tstep

#  Trial solution function construction
psi_L(x) = (b - x) / x_domain       # left
psi_R(x) = (x - a) / x_domain       # right
phi_B(t) = (h - t) / h   # bottom

function T1NN(t, x, tn, params)
    return psi_L(x) * PNN([t, a], params)[1] +
           psi_R(x) * PNN([t, b], params)[1] +
           phi_B(t) * PNN([0.0, x], params)[1]
end

function T2NN(t, x, tn, params)
    return psi_L(x) * phi_B(t) * PNN([0.0, a], params)[1] +
           psi_R(x) * phi_B(t) * PNN([0.0, b], params)[1]
end

function C1(t, x, tn, params)
    return psi_L(x) * exact_u(t, a) +
           psi_R(x) * exact_u(t, b) +
           phi_B(t) * exact_u(0, x)
end

function C2(t, x, tn, params)
    return psi_L(x) * phi_B(t) * exact_u(0, a) +
           psi_R(x) * phi_B(t) * exact_u(0, b)
end

function u_trial(t, x, tn, params)
    PNN([t, x], params)[1] - T1NN(t, x, tn, params) + T2NN(t, x, tn, params) +
    C1(t, x, tn, params) - C2(t, x, tn, params)
end
v_trial(t, x, tn, params) = Zygote.gradient(tt -> u_trial(tt, x, tn, params), t)[1]
w_trial(t, x, tn, params) = Zygote.gradient(xx -> u_trial(t, xx, tn, params), x)[1]
# @benchmark (v_trial(0.3+1e-5, -2.0, 0.3, PNN.params) - v_trial(0.3-1e-5, -2.0, 0.3, PNN.params)) / (2*1e-5)

u_trial(0.3, -2.0, 0.3, PNN.params)
PNN([0.3, -2.0], PNN.params)[1]
T1NN(0.3, -2.0, 0.3, PNN.params)
T2NN(0.3, -2.0, 0.3, PNN.params)
C1(0.3, -2.0, 0.3, PNN.params)
C2(0.3, -2.0, 0.3, PNN.params)

utt_trial(t, x, tn, params) = Zygote.gradient(tt -> v_trial(tt, x, tn, params), t)[1]
uxx_trial(t, x, tn, params) = Zygote.gradient(xx -> w_trial(t, xx, tn, params), x)[1]
utt_trial(0.3, -2.0, 0.3, PNN.params)

utt_jac(t, x, tn, params) = Zygote.hessian(tt -> u_trial(tt, x, tn, params), t)[1]
utt_jac(0.3, -2.0, 0.3, PNN.params)

uxx_jac(t, x, tn, params) = Zygote.hessian(xx -> u_trial(t, xx, tn, params), x)[1]
uxx_jac(0.3, -2.0, 0.3, PNN.params)

# v_trial(t,x,tn,params) = ForwardDiff.derivative(tt -> u_trial(tt,x,tn,params),t)
# v_trial(0.3, -2.0, 0.3, PNN.params)

# w_trial(t,x,tn,params) = ForwardDiff.derivative(tt -> u_trial(tt,x,tn,params),t)

# ∂u∂θ_func(t,x,tn,params) = Zygote.gradient(ps -> u_trial(t,x,tn,ps), params)[1]
# ∂v∂θ_func(t,x,tn,params) = Zygote.gradient(ps -> v_trial(t,x,tn,ps), params)[1]
# ∂w∂θ_func(t,x,tn,params) = Zygote.gradient(ps -> w_trial(t,x,tn,ps), params)[1]
# u_trial(0.3, -2.0, 0.3, PNN.params)
# ∂u∂θ_func(0.3, -2.0, 0.3, PNN.params)

t_val = collect(tspan[1]:0.01:tspan[2])
x_val = collect(xspan[1]:0.01:xspan[2])

Z = [u_trial(t, x, 0.0, PNN.params) for (t, x) in Iterators.product(t_val, x_val)]
analytic_sol = [exact_u(t, x) for (t, x) in Iterators.product(t_val, x_val)]
Z .- analytic_sol

tx_in = rand(Random.seed!(1), 2, 5000)
tx_in[2, :] .= xspan[1] .+ (xspan[2] - xspan[1]) * tx_in[2, :]

# PINN loss for Linear Transport Equation: u_t + u_x = 0
# function pinn_loss(params, tx_in)
#     loss = 0.0
#     for i in 1:size(tx_in, 2)
#         t, x = tx_in[:, i]
#         ut = v_trial(t, x, 0.0, params)
#         ux = w_trial(t, x, 0.0, params)
#         res = ut + 0.2 * ux
#         loss += res^2
#     end
#     return loss / size(tx_in, 2)
# end
# MSE loss function for supervised training
function mse_loss(params, tx_in)
    loss = 0.0
    for i in 1:size(tx_in, 2)
        t, x = tx_in[:, i]
        pred = u_trial(t, x, 0.0, params)
        label = exact_u(t, x)
        loss += (pred - label)^2
    end
    return loss / size(tx_in, 2)
end

# Training loop for supervised learning
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

        grads = Zygote.gradient(d -> mse_loss(d, batch_tx), PNN.params)[1]
        GeometricMachineLearning.optimization_step!(opt, λ, PNN.params, grads)
        batch_loss = mse_loss(PNN.params, batch_tx)
        epoch_loss += batch_loss * size(batch_tx, 2)
    end
    epoch_loss /= num_samples
    push!(loss_history, epoch_loss)
    println("Epoch $epoch, MSE Loss: $epoch_loss")
end
# Training loop
# opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.GradientOptimizer(.001), PNN.params)
# λ = GeometricMachineLearning.GlobalSection(PNN.params)

# loss_history = []

# batch_size = 100
# num_samples = size(tx_in, 2)
# num_batches = cld(num_samples, batch_size)

# # for epoch in 1:500
# #     epoch_loss = 0.0
#     # for batch_idx in 1:num_batches
#     batch_idx = 1
#     batch_start = (batch_idx - 1) * batch_size + 1
#     batch_end = min(batch_idx * batch_size, num_samples)
#     batch_tx = tx_in[:, batch_start:batch_end]

#     grads = ForwardDiff.gradient(d -> pinn_loss(d, batch_tx), PNN.params)[1]
#     GeometricMachineLearning.optimization_step!(opt, λ, PNN.params, grads)

#     batch_loss = pinn_loss(PNN.params, batch_tx)
#     epoch_loss += batch_loss * size(batch_tx, 2)
# #     end
# #     epoch_loss /= num_samples
# #     push!(loss_history, epoch_loss)
# #     println("Epoch $epoch, Loss: $epoch_loss")

# # end

# # Plot loss history
# plot(loss_history, xlabel="Epoch", ylabel="PINN Loss", title="Training Loss History")

t_vals = collect(tspan[1]:0.01:tspan[2])
x_vals = collect(xspan[1]:0.01:xspan[2])
Z_pred = [u_trial(t, x, 0.0, PNN.params) for t in t_vals, x in x_vals]

heatmap(
    x_vals, t_vals, Z_pred,
    xlabel = "x", ylabel = "t", title = "Neural Network Prediction",
    colorbar_title = "u_pred"
)

Z_exact = [exact_u(t, x) for t in t_vals, x in x_vals]

heatmap(
    x_vals, t_vals, Z_exact,
    xlabel = "x", ylabel = "t", title = "Analytic Solution",
    colorbar_title = "u_exact"
)

heatmap(
    x_vals, t_vals, Z_pred .- Z_exact,
    xlabel = "x", ylabel = "t", title = "Error (u_pred - u_exact)",
    colorbar_title = "Error"
)

N_in = 500
tx_in = rand(Random.seed!(1), 2, N_in)
tx_in[2, :] .= xspan[1] .+ (xspan[2] - xspan[1]) * tx_in[2, :]
intermidiate_ps = (L1 = PNN.params.L1, L2 = PNN.params.L2)
AbstractNeuralNetworks.Chain(PNN.model.layers[1:(end - 1)]...)([0.0, 2.0], intermidiate_ps)

u0 = PNN.params.L3.W[:]
# prob = NonlinearLeastSquaresProblem(
# NonlinearFunction(nlls!, resid_prototype = zeros(N_in)), u0, PNN)
# u_sol = solve(prob,maxtime = 60,abstol = 1e-12, reltol = 1e-12).u

u_basis_nn = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh)
)

u_basis_func = NeuralNetwork(u_basis_nn)
u0 = PNN.params.L3.W[:]
u_basis_func.params.L1.W .= PNN.params.L1.W
u_basis_func.params.L1.b .= PNN.params.L1.b
u_basis_func.params.L2.W .= PNN.params.L2.W
u_basis_func.params.L2.b .= PNN.params.L2.b

function T1NN2(t, x, tn, dofs)
    return psi_L(x) * sum(dofs .* u_basis_func([t, a])) +
           psi_R(x) * sum(dofs .* u_basis_func([t, b])) +
           phi_B(t) * sum(dofs .* u_basis_func([0.0, x]))
end

function T2NN2(t, x, tn, dofs)
    return psi_L(x) * phi_B(t) * sum(dofs .* u_basis_func([0.0, a])) +
           psi_R(x) * phi_B(t) * sum(dofs .* u_basis_func([0.0, b]))
end

function C12(t, x, tn, dofs)
    return psi_L(x) * exact_u(t, a) +
           psi_R(x) * exact_u(t, b) +
           phi_B(t) * exact_u(0, x)
end

function C22(t, x, tn, dofs)
    return psi_L(x) * phi_B(t) * exact_u(0, a) +
           psi_R(x) * phi_B(t) * exact_u(0, b)
end

function u_trial2(t, x, tn, dofs)
    sum(dofs .* u_basis_func([t, x])) - T1NN2(t, x, tn, dofs) + T2NN2(t, x, tn, dofs) +
    C12(t, x, tn, dofs) - C22(t, x, tn, dofs)
end
v_trial2(t, x, tn, dofs) = Zygote.gradient(tt -> u_trial2(tt, x, tn, dofs), t)[1]
w_trial2(t, x, tn, dofs) = Zygote.gradient(xx -> u_trial2(t, xx, tn, dofs), x)[1]

utt(t, x, tn, dofs) = Zygote.hessian(tt -> u_trial2(tt, x, tn, dofs), t)[1]
uxx(t, x, tn, dofs) = Zygote.hessian(xx -> u_trial2(t, xx, tn, dofs), x)[1]
utt(0.3, -2.0, 0.3, u0)
uxx(0.3, -2.0, 0.3, u0)

sum(u0 .* u_basis_func([0.3, -2.0]))
T1NN2(0.3, -2.0, 0.3, u0)
T2NN2(0.3, -2.0, 0.3, u0)
C12(0.3, -2.0, 0.3, u0)
C22(0.3, -2.0, 0.3, u0)

function nlls!(du, u, p)
    for i in 1:N_in
        du[i] = utt(tx_in[1, i], tx_in[2, i], 0.0, u) -
                uxx(tx_in[1, i], tx_in[2, i], 0.0, u)
    end
end

prob = NonlinearLeastSquaresProblem(
    NonlinearFunction(nlls!, resid_prototype = zeros(N_in)), u0)
u_sol = solve(prob, maxtime = 60, abstol = 1e-12, reltol = 1e-12).u
