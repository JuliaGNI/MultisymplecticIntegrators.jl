using GeometricIntegrators
using QuadratureRules
using MultiSymplectic
using AbstractNeuralNetworks
using Random
using GeometricMachineLearning
using Zygote
using Plots
using ForwardDiff

u_network = Chain(
    Dense(2, 20, tanh),
    Dense(20, 10, tanh)
)

nn_elm_basis = ELM_NN_Basis(u_network, 10)
elm_int = ELM_PDE_int(nn_elm_basis; RT=8,RX = 16)
lpde = MultiSymplectic.LinearTransport.lpdeproblem(tstep = 0.2,tspan =(0.0,1.0))
sol = MultiSymplectic.integrate(lpde,elm_int)