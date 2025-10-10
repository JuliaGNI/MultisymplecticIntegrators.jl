using MultiSymplectic
using AbstractNeuralNetworks
using Plots
using Profile
lpde = MultiSymplectic.LinearTransport.lpdeproblem(tstep = 0.3,tspan =(0.0,0.6),xspan = (-4.,-1.))

NP = 50
basis_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh,),
    Dense(100, 100, tanh,),
    Dense(100, NP, tanh),
)

PNN_basis = NeuralNetwork(basis_network)

u_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, NP, tanh),
    Dense(NP, 1,identity,use_bias = false),
)

u_func = NeuralNetwork(u_network)
trial_basis = Trial_Solution_Basis(PNN_basis,u_func,NP)
trial_int = TrialNN_PDE_int(trial_basis)

trial_sol = MultiSymplectic.integrate(lpde,trial_int)
