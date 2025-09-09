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
    Dense(2, 100, tanh),
    Dense(100, 20, tanh)
)

nn_elm_basis = ELM_NN_Basis(u_network, 20)
elm_int = ELM_PDE_int(nn_elm_basis; RT=4,RX = 12)
lpde = MultiSymplectic.LinearTransport.lpdeproblem(tstep = 0.1,tspan =(0.0,1.0))
sol = MultiSymplectic.integrate(lpde,elm_int)