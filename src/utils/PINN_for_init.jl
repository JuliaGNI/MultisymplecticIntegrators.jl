using AbstractNeuralNetworks
using GeometricMachineLearning
using MultiSymplectic
using LinearAlgebra
using Statistics
using Zygote
using IterTools
function box_init_resnet!(PNN)
    m = 1.0
    L = length(keys(PNN.params))
    # for l in 2:L
    for (l, (name, layer)) in enumerate(zip(keys(PNN.params), values(PNN.params)))
        if l ==1
            # First layer
            width = size(layer[keys(layer)[1]])[1]
            input_dim = size(layer[keys(layer)[1]])[2]
            layer.W[:],layer.b[:] = MultiSymplectic.box_init_plain(input_dim, width)
            continue
        end
        if l == L
            # Last layer
            width = size(layer[keys(layer)[1]])[2]
            layer.W[:], _= MultiSymplectic.box_init_plain(width, 1)
            continue
        end
        width = size(layer[keys(layer)[1]])[2]
        m *= (1 + 1 / (L - 1))
        W = zeros(Float32, width, width)
        b = zeros(Float32, width)
        for i in 1:width
            p = m * rand(Float32, width)
            n = randn(Float32, width)
            n ./= LinearAlgebra.norm(n)
            p_max = map(nj -> nj ≥ 0 ? m : 0.0f0, n)
            k = 1 / (dot(p_max .- p, n) * (L - 1))
            W[i, :] = k * n
            b[i] = k * dot(p, n)
        end
        layer[keys(layer)[1]][:] = W
        layer[keys(layer)[2]][:] = b
    end
end


u_network = Chain(
    Dense(2, 32, tanh),
    # GeometricMachineLearning.ResNetLayer(24, tanh),
    Dense(32, 1,identity,use_bias = false)
)

PNN = NeuralNetwork(u_network)
box_init_resnet!(PNN)


# --------------------------------------------------
# PINN Loss Function
# --------------------------------------------------
velocity(x, t) = x  # a(x, t) = x
u0(x) = max(0, 1 - abs(2x - 1))  # initial condition
analytic_solution(t,x) = u0(x-t)
function pinn_loss(int_pts, init_pts, bc_pts, arch,params; ε=1.0)
    t_int, x_int = int_pts
    t0, x0 = init_pts
    tb, xb = bc_pts

    J1 = Statistics.mean([
        let input = hcat(t, x)'  # time first
            ∂u = gradient(tx -> arch(tx, params)[1],input)[1]
            # ∂u∂x = Zygote.gradient(x -> arch(hcat(t, x)',params)[1], x)[1]
            # ∂u∂t = Zygote.gradient(tt -> arch(hcat(tt, x)',params)[1], t)[1]
            ∂u∂x = ∂u[2]
            ∂u∂t = ∂u[1]
            r = ∂u∂t + velocity(x, t) * ∂u∂x
            r^2
        end for (t, x) in zip(t_int, x_int)
    ])

    J2 = Statistics.mean([(arch(hcat(0.0, x)',params)[1] - u0(x))^2 for x in x0])
    J3 = Statistics.mean([(arch(hcat(t, 0.0)',params)[1])^2 for t in tb])

    return ε * J1 + J2 + J3
end


dx = 0.05
x_int = collect(0:dx:1.)
t_int = collect(0:dx:1.0)
grid = collect(IterTools.product(t_int, x_int))
grid = vcat(grid...)  # Convert to a matrix with time in the first row and space in the second row
tx_int = hcat(collect.(grid)...)'
    
# tx_int = [t_int x_int]
t0 = zeros(Float32, size(x_int))
x0 = x_int  # Initial condition at t=0
tx_0 = [t0 x_int]

xb = zeros(Float32, size(t_int))
tb = t_int  # Boundary condition at t=1
tx_b = [t_int xb]


bc_initial_inputs = vcat(tx_0, tx_b)'
bc_labels = analytic_solution.(bc_initial_inputs[1,:], bc_initial_inputs[2,:])
bc_labels = reshape(bc_labels, 1, :)  # labels should be a matrix with one row and multiple columns
# u_network(network_inputs, PNN.params)


data = ((tx_int[:, 1], tx_int[:, 2]), (t0, x0), (tb, xb))
# pinn_loss((t_int, x_int), (t0, x0), (tb, xb), u_network, PNN.params; ε=1.0)
# Zygote.gradient(p -> pinn_loss((t_int, x_int), (t0, x0), (tb, xb), u_network, p; ε=1.0), PNN.params)


tem_ps = PNN.params[keys(PNN.params)[1:end-1]]
opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.GradientOptimizer(.005), tem_ps)
λ = GeometricMachineLearning.GlobalSection(tem_ps)

nepochs = 20
err_ls = zeros(Float64, nepochs)
for ep in 1:nepochs
    Φ = AbstractNeuralNetworks.Chain(u_network.layers[1:end-1]...)(bc_initial_inputs,tem_ps)
    # Φ = NN(network_inputs, PNN.params)
    # PNN.params.L3.W[:] = labels/Φ
    PNN.params[keys(PNN.params)[end]].W[:] = (Φ' \ bc_labels')'
    gs = Zygote.gradient(p -> pinn_loss((tx_int[:, 1], tx_int[:, 2]), (t0, x0), (tb, xb), u_network, p; ε=1.0), PNN.params)[1]
    tem_ps = PNN.params[keys(PNN.params)[1:end-1]]
    tem_gs = gs[keys(gs)[1:end-1]]
    GeometricMachineLearning.optimization_step!(opt,λ, tem_ps, tem_gs)
    err_ls[ep] = pinn_loss((tx_int[:, 1], tx_int[:, 2]), (t0, x0), (tb, xb), u_network, PNN.params; ε=1.0)
    println("Epoch: $ep, Loss: $(err_ls[ep])")
    @show PNN.params
end

plot(err_ls, title = "Loss over epochs", xlabel = "Epochs", ylabel = "Loss")






using Plots
plot(u0.(-1:0.01:1))

test_x_ls = collect(0:0.01:10.)
test_t_ls = collect(0:0.01:10.0)
test_grid = collect(IterTools.product(test_t_ls, test_x_ls))
truth_sol = zeros(size(test_grid))
for i in 1:size(test_grid, 1)
    for j in 1:size(test_grid, 2)
        truth_sol[i,j] = analytic_solution(test_grid[i,j][1], test_grid[i,j][2])
    end
end
plot(test_t_ls, test_x_ls,truth_sol)


