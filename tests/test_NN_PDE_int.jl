using GeometricIntegrators
using QuadratureRules
using MultiSymplectic
using AbstractNeuralNetworks
using Random
using GeometricMachineLearning
using Zygote
# by default, the first dimension is time, the second is space, 
# and just consider 1+1 now. 
relu2(x) = max.(0, x) ^2
u_network = Chain(
    Dense(2, 8, relu2),
    Dense(8, 1,identity,use_bias = false)
)

# GeometricIntegrators.Integrators.default_options(::NN_PDE_Integrator) = Options(
#     x_reltol = 8eps(),
#     x_suctol = 2eps(),
#     f_abstol = 8eps(),
#     f_reltol = 8eps(),
#     f_suctol = 2eps(),
#     max_iterations = 10_000,
# )

pnn = NeuralNetwork(u_network)
nn_pde_basis = NetworkPDEBasis(u_network,:Fully)
t_step = 0.1
x_span = (0.,1.)
nn_int = NN_PDE_Integrator(nn_pde_basis,RT = 4,RX = 6, xspan = x_span, tstep = t_step,μ =:BSplineDirichlet,λ =:BSplineDirichlet,
                            k_μ = 4,k_λ₀_x = 4,nepochs= 2,initial_guess_method = LSGD(),params_turbulance = 0.001)
lpde = MultiSymplectic.LinearTransport.lpdeproblem(tstep = t_step,tspan =(0.0,0.1),xspan = x_span)
sol = MultiSymplectic.integrate(lpde,nn_int)

lpde.exact_u.(0.1,collect(0:0.01:0.5))
plot(lpde.exact_u.(0.1,collect(-1.5:0.01:1.5)))
plot!(sol.sol.u[1])
