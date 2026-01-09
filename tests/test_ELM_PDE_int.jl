using QuadratureRules
using MultiSymplectic
using AbstractNeuralNetworks
using Random
using GeometricMachineLearning
using Zygote
using Plots
using ForwardDiff
using Infiltrator

Random.seed!(123)
S = 200
u_network = Chain(
    Dense(2, S, tanh),
    Dense(S, S, tanh),
    Dense(S, S, tanh),
)

nn_elm_basis = NN_Basis(u_network, S)

xspan = (0.0,1.0)
elm_int = ELM_PDE_int(nn_elm_basis; RT=18,RX = 18,xspan = xspan,initial_guess_method = ELM(), show_status=true)
lpde = MultiSymplectic.Wave.lpdeproblem(timestep=0.3, timespan =(0.0,0.3),xspan = xspan)
sol = MultiSymplectic.integrate(lpde,elm_int)

