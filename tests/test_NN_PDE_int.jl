using QuadratureRules
using MultiSymplectic
using AbstractNeuralNetworks
using Random
using GeometricMachineLearning
using Zygote
using Plots
using ForwardDiff
using Base

using Infiltrator
using Base
using GeometricIntegratorsBase
using Plots
# by default, the first dimension is time, the second is space, 
# and just consider 1+1 now. 
relu2 = x -> max(0, x) ^2
relu3 = x -> max(0, x) ^3

GeometricIntegratorsBase.default_options(::NN_PDE_Integrator) = (
    x_abstol = 8eps(),
    f_abstol = 8eps(),
    max_iterations = 100,
)

S = 150
nn_pde_basis = NetworkPDEBasis(S,relu3,:Partially) # Partially, Fully
xspan = (0.0,1.0)

nn_int = NN_PDE_Integrator(nn_pde_basis,RT = 64,RX = 64, xspan = xspan, μ =:BSplineDirichlet,λ =:BSplineDirichlet,
k_μ_t = 4,k_λ_x = 4,nepochs= 1,initial_guess_method = OGA2D(), show_status=true)

t_step = 0.3
lpde = MultiSymplectic.Wave.lpdeproblem(timestep=0.3, timespan =(0.0,t_step),xspan = xspan)

log_file="logs/NN_pde.txt"
open(log_file, "w") do io
    redirect_stdout(io) do
        sol = MultiSymplectic.integrate(lpde,nn_int)
       
        p = @layout [a b c]
        p1 = plot([lpde.exact_u(0.3,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_u")
        plot!(p1, sol.u[1], label="sol.u")
        p2 = plot([lpde.exact_v(0.3,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_v")
        plot!(p2, sol.v[1], label="sol.v")
        p3 = plot([lpde.exact_w(0.3,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_w")
        plot!(p3, sol.w[1], label="sol.w")
        p = plot(p1, p2, p3, layout=p)
        savefig("logs/NNInt_t=h.pdf")

        # p = @layout [a b c]
        # p1 = plot([lpde.exact_u(0.6,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_u")
        # plot!(p1, sol.u[2], label="sol.u")
        # p2 = plot([lpde.exact_v(0.6,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_v")
        # plot!(p2, sol.v[2], label="sol.v")
        # p3 = plot([lpde.exact_w(0.6,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_w")
        # plot!(p3, sol.w[2], label="sol.w")
        # p = plot(p1, p2, p3, layout=p)
        # savefig("logs/NNInt_t=2h.pdf")
#         savefig("logs/NN_pde.pdf")
    end
end


S = 100
u_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, S, tanh)
)

nn_elm_basis = NN_Basis(u_network, S)