using MultiSymplectic
using AbstractNeuralNetworks
using Plots
using Profile
lpde = MultiSymplectic.LinearTransport.lpdeproblem(tstep = 0.3,tspan =(0.0,0.6),xspan = (-4.,-1.))

u_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh,),
    Dense(100, 100, tanh,),
    Dense(100, 50, tanh,use_bias = false),
)

PNN = NeuralNetwork(u_network)
trial_basis = Trial_Solution_Basis(PNN,10)
trial_int = TrialNN_PDE_int(trial_basis)

@profile MultiSymplectic.integrate(lpde,trial_int)