# using GeometricIntegrators
using QuadratureRules
using MultiSymplectic
using AbstractNeuralNetworks
using Random
using GeometricMachineLearning
using Zygote
using Plots
using ForwardDiff

Random.seed!(123)
S = 100
u_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, S, tanh)
)

nn_elm_basis = NN_Basis(u_network, S)
elm_int = ELM_PDE_int(nn_elm_basis; RT=12,RX = 32)
lpde = MultiSymplectic.LinearTransport.lpdeproblem(tstep = 0.3,tspan =(0.0,1.5),xspan = (-4.,-1.))
sol = MultiSymplectic.integrate(lpde,elm_int)
# sol.sol.u[2]
plot([MultiSymplectic.LinearTransport.exact_u(0.1,x) for x in -4.:0.01:-1.],label = "Exact at t=0.1")
plot!(sol.sol.u[1],label = "ELM_NN at t=0.1")
    

plot([MultiSymplectic.LinearTransport.exact_u(0.2,x) for x in -4.:0.01:-1.],label = "Exact at t=0.2")
plot!(sol.sol.u[2],label = "ELM_NN at t=0.2")

plot([MultiSymplectic.LinearTransport.exact_u(0.3,x) for x in -4.:0.01:-1.],label = "Exact at t=0.3")
plot!(sol.sol.u[3],label = "ELM_NN at t=0.3")

plot([MultiSymplectic.LinearTransport.exact_u(0.4,x) for x in -4.:0.01:-1.],label = "Exact at t=0.4")
plot!(sol.sol.u[4],label = "ELM_NN at t=0.4")

plot([MultiSymplectic.LinearTransport.exact_u(0.5,x) for x in -4.:0.01:-1.],label = "Exact at t=0.5")
plot!(sol.sol.u[5],label = "ELM_NN at t=0.5")

err = [maximum((sol.sol.u[i] .- [MultiSymplectic.LinearTransport.exact_u(0.1*i,x) for x in -4.:0.01:-1.]) ./ [MultiSymplectic.LinearTransport.exact_u(0.1*i,x) for x in -4.:0.01:-1.]) for i in 1:5]