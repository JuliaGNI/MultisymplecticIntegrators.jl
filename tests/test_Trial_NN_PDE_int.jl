using MultiSymplectic
using AbstractNeuralNetworks
using Plots

lpde = MultiSymplectic.LinearTransport.lpdeproblem(tstep = 0.3,tspan =(0.0,1.5),xspan = (-4.,-1.))

u_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 1, tanh)
)

PNN = NeuralNetwork(u_network)
trial_basis = Trial_Solution_Basis(PNN,lpde,:TFC)
exact_u = MultiSymplectic.LinearTransport.exact_u
trial_basis.u(0.1, 0.25, 0.0, PNN.params)

h = 0.3
xspan = (-4.0, -1.0)
a = xspan[1]
b = xspan[2]
x_length = b - a


t_vals = range(0.0, 0.3, length=50)
x_vals = range(-4.0, -1.0, length=100)

analytic_sol = [MultiSymplectic.LinearTransport.exact_u(t,x) for t in t_vals, x in x_vals]
Z = [trial_basis.u(t, x, 0.0, PNN.params) for t in t_vals, x in x_vals]
heatmap(x_vals, t_vals, analytic_sol .- Z, xlabel="x", ylabel="t")


