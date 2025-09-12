using MultiSymplectic
using AbstractNeuralNetworks
lpde = MultiSymplectic.LinearTransport.lpdeproblem(tstep = 0.3,tspan =(0.0,1.5),xspan = (-4.,-1.))

u_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 1, tanh)
)
PNN = NeuralNetwork(u_network)
trial_basis = Trial_Solution_Basis(PNN,lpde,:TFC)
trial_basis.u(0.1, 0.25, 0.0, PNN.params)
using Plots

t_vals = range(0.0, 0.3, length=100)
x_vals = range(-4.0, -1.0, length=100)


Z = [trial_basis.u(t, x, 0.0, PNN.params) for t in t_vals, x in x_vals]
heatmap(x_vals, t_vals, Z, xlabel="x", ylabel="t", title="Trial Basis Value")