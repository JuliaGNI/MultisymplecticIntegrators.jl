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
