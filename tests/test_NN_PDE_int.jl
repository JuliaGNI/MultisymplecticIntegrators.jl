using GeometricIntegrators
using QuadratureRules
using MultiSymplectic
using AbstractNeuralNetworks
using Random
using GeometricMachineLearning
using Zygote
# by default, the first dimension is time, the second is space, 
# and just consider 1+1 now. 
u_network = Chain(
    Dense(2, 24, tanh),
    Dense(24, 1,identity,use_bias = false)
)

GeometricIntegrators.Integrators.default_options(::NN_PDE_Integrator) = Options(
    x_reltol = 8eps(),
    x_suctol = 2eps(),
    f_abstol = 8eps(),
    f_reltol = 8eps(),
    f_suctol = 2eps(),
    max_iterations = 10_000,
)

pnn = NeuralNetwork(u_network)
nn_pde_basis = NetworkPDEBasis(u_network)
t_step = 0.1
x_span = (-1.,1.)
nn_int = NN_PDE_Integrator(nn_pde_basis,RT = 4,RX = 32, xspan = x_span, tstep = t_step,μ =:BSplineDirichlet,λ =:BSplineDirichlet,k_μ = 4,k_λ₀_x = 4,nepochs= 1000)
lpde = MultiSymplectic.SineGordon.lpdeproblem(tstep = t_step,tspan =(0.0,0.1),xspan = x_span)
sol = MultiSymplectic.integrate(lpde,nn_int)




lpde.exact_u.(0.1,collect(0:0.01:0.5))
plot(lpde.exact_u.(0.1,collect(-1.5:0.01:1.5)))
plot!(sol.sol.u[1])




using Plots
using IterTools: product


RT = 4
RX = 32
x_span = (-1.5,1.5)
x_domain  = x_span[2] - x_span[1]
network_inputs,_ = MultiSymplectic.construct_quadrature_grid_with_boundary([RT,RX])
labels = MultiSymplectic.SineGordon.exact_u.(t_step*network_inputs[1,:],x_span[1] .+ x_domain .* network_inputs[2,:])
labels = reshape(labels,1,:) # labels should be a matrix with one row and multiple columns

NN = Chain(
    Dense(2, 50, tanh),
    Dense(50, 1,identity,use_bias = false)
)
PNN = NeuralNetwork(NN)


# initialize the parameters and train with LSGD
for (name, layer) in zip(keys(PNN.params), values(PNN.params) )
    in_size = size(layer.W, 2)
    out_size = size(layer.W, 1)
    if hasfield(typeof(layer), :b)
        layer.W[:], layer.b[:] = MultiSymplectic.box_init_plain(in_size, out_size)
    else
        # For layers without bias (e.g., output), just regenerate W
        layer.W[:], _ = MultiSymplectic.box_init_plain(in_size, out_size)
    end
end

tem_ps = PNN.params[keys(PNN.params)[1:end-1]]
opt = GeometricMachineLearning.Optimizer(GeometricMachineLearning.GradientOptimizer(.0001), tem_ps)
err = 0
λ = GeometricMachineLearning.GlobalSection(tem_ps)
nepochs = 2000
err_ls = zeros(Float64, nepochs)
for ep in 1:nepochs
    Φ = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)(network_inputs,tem_ps)
    # Φ = NN(network_inputs, PNN.params)
    # PNN.params.L3.W[:] = labels/Φ
    PNN.params[keys(PNN.params)[end]].W[:] = (Φ' \ labels')'
    gs = Zygote.gradient(p -> MultiSymplectic.lsgd_loss(network_inputs,labels,NN,p),PNN.params)[1]
    tem_ps = PNN.params[keys(PNN.params)[1:end-1]]
    tem_gs = gs[keys(gs)[1:end-1]]
    GeometricMachineLearning.optimization_step!(opt,λ, tem_ps, tem_gs)
    err_ls[ep] = MultiSymplectic.lsgd_loss(network_inputs,labels,NN,PNN.params)
end
plot(err_ls, label = "Loss", xlabel = "Epochs", ylabel = "Loss", title = "Training Loss", size = (800, 400))

# Prepare test grid
test_x_ls = collect(0:0.01:1.)
test_t_ls = collect(0:0.01:0.1)
test_grid = collect(product(test_t_ls, test_x_ls))
#Flatten the grid to match the input shape of the network
NN_pred = zeros(size(test_grid))
truth_sol = zeros(size(test_grid))
for i in 1:size(test_grid, 1)
    for j in 1:size(test_grid, 2)
        NN_pred[i,j] = PNN([test_grid[i,j][1],test_grid[i,j][2]], PNN.params)[1]
        truth_sol[i,j] = lpde.exact_u(test_grid[i,j][1], test_grid[i,j][2])
    end
end

surface(test_t_ls, test_x_ls, NN_pred', xlabel = "x", ylabel = "t", zlabel = "u", title = "NN Prediction", size = (800, 400))
surface(test_t_ls, test_x_ls, truth_sol', xlabel = "x", ylabel = "t", zlabel = "u", title = "Exact Solution", size = (800, 400))

surface(test_t_ls, test_x_ls, NN_pred' - truth_sol', xlabel = "x", ylabel = "t", zlabel = "Error", title = "Prediction Error", size = (800, 400))