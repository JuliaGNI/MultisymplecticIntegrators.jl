using GeometricIntegrators
using QuadratureRules
using MultiSymplectic
using AbstractNeuralNetworks
using Random

# by default, the first dimension is time, the second is space, 
# and just consider 1+1 now. 
u_network = Chain(
    Dense(2, 50, tanh),
    Dense(50, 1,identity,use_bias = false)
)

GeometricIntegrators.Integrators.default_options(::NN_PDE_Integrator) = Options(
    x_reltol = 8eps(),
    x_suctol = 2eps(),
    f_abstol = 8eps(),
    f_reltol = 8eps(),
    f_suctol = 2eps(),
    max_iterations = 10_000,
)

pnn = NeuralNetwork(u_network)
nn_pde_basis = NetworkPDEBasis(u_network)
t_step = 0.1
x_span = (0.,0.5)
nn_int = NN_PDE_Integrator(nn_pde_basis,RT = 4,RX = 8, xspan = x_span, tstep = t_step,μ =:BSplineDirichlet,λ =:BSplineDirichlet,k_μ = 4,k_λ₀_x = 4,nepochs= 1000)
lpde = MultiSymplectic.SineGordon.lpdeproblem(tstep = t_step,tspan =(0.0,1.0),xspan = x_span)
sol = MultiSymplectic.integrate(lpde,nn_int)
