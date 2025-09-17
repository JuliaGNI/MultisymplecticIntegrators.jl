using AbstractNeuralNetworks
using GeometricMachineLearning
using MultiSymplectic
using LinearAlgebra
using Statistics
using Zygote
using IterTools
using Plots
using Random
u_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100,tanh,),
    Dense(100, 1,identity,use_bias = false),
)

PNN = NeuralNetwork(u_network)
exact_u = MultiSymplectic.LinearTransport.exact_u
exact_u(0.0,0.5)

tstep = 0.3
tspan =(0.0,1.5)
xspan = (-4.,-1.)
a,b = xspan[1], xspan[2]
x_domain = xspan[2] - xspan[1]
h = tstep

#  Trial solution function construction
psi_L(x) = (b - x) / x_domain       # left 
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
    return psi_L(x) * exact_u(t,a)  +
        psi_R(x) * exact_u(t,b) +
        phi_B(t) * exact_u(0,x) 

end

function C2(t, x, tn, params)
    return psi_L(x) * phi_B(t) * exact_u(0,a) +
        psi_R(x) * phi_B(t) * exact_u(0,b)
end


u_trial(t,x,tn,params) = PNN([t,x])[1] - T1NN(t,x,tn,params) + T2NN(t,x,tn,params) + C1(t,x,tn,params) - C2(t,x,tn,params)
v_trial(t,x,tn,params) = Zygote.gradient(tt -> u_trial(tt,x,tn,params),t)[1]
w_trial(t,x,tn,params) = Zygote.gradient(xx -> u_trial(t,xx,tn,params),x)[1]


∂u∂θ_func(t,x,tn,params) = Zygote.gradient(ps -> u_trial(t,x,tn,ps), params)[1]
∂v∂θ_func(t,x,tn,params) = Zygote.gradient(ps -> v_trial(t,x,tn,ps), params)[1]
∂w∂θ_func(t,x,tn,params) = Zygote.gradient(ps -> w_trial(t,x,tn,ps), params)[1]
u_trial(0.3, -2.0, 0.3, PNN.params)
∂u∂θ_func(0.3, -2.0, 0.3, PNN.params)

# t_val = collect(tspan[1]:0.01:tspan[2])
# x_val = collect(xspan[1]:0.01:xspan[2])

# Z = [u_trial(t,x,0.0, PNN.params) for (t,x) in Iterators.product(t_val,x_val)]
# analytic_sol = [exact_u(t,x) for (t,x) in Iterators.product(t_val,x_val)]
# Z .- analytic_sol

tx_in = rand(Random.seed!(1),2,2000)
tx_in[2,:] .= xspan[1] .+ (xspan[2] - xspan[1]) * tx_in[2,:]

# PINN loss for Linear Transport Equation: u_t + u_x = 0
function pinn_loss(dofs, tx_in)
    loss = 0.0
    for i in 1:size(tx_in, 2)
        t, x = tx_in[:, i]
        ut = v_trial(t, x, 0.0, dofs)
        ux = w_trial(t, x, 0.0, dofs)
        res = ut + 0.2 * ux
        loss += res^2
    end
    return loss / size(tx_in, 2)
end

# Training loop
opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.GradientOptimizer(.001), PNN.params)
λ = GeometricMachineLearning.GlobalSection(PNN.params)

loss_history = []

batch_size = 100
num_samples = size(tx_in, 2)
num_batches = cld(num_samples, batch_size)

# for epoch in 1:500
#     epoch_loss = 0.0
    # for batch_idx in 1:num_batches
    batch_idx = 1
    batch_start = (batch_idx - 1) * batch_size + 1
    batch_end = min(batch_idx * batch_size, num_samples)
    batch_tx = tx_in[:, batch_start:batch_end]

    grads = Zygote.gradient(d -> pinn_loss(d, batch_tx), PNN.params)[1]
    GeometricMachineLearning.optimization_step!(opt, λ, PNN.params, grads)

    batch_loss = pinn_loss(PNN.params, batch_tx)
    epoch_loss += batch_loss * size(batch_tx, 2)
#     end
#     epoch_loss /= num_samples
#     push!(loss_history, epoch_loss)
#     println("Epoch $epoch, Loss: $epoch_loss")
    
# end

# Plot loss history
plot(loss_history, xlabel="Epoch", ylabel="PINN Loss", title="Training Loss History")