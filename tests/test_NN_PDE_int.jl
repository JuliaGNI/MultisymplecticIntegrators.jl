using GeometricIntegrators
using QuadratureRules
using MultiSymplectic
using AbstractNeuralNetworks
using Random
using GeometricMachineLearning
using Zygote
using Plots
using ForwardDiff
using Base
# by default, the first dimension is time, the second is space, 
# and just consider 1+1 now. 
relu2 = x -> max(0, x) ^2
relu3 = x -> max(0, x) ^3


# GeometricIntegrators.Integrators.default_options(::NN_PDE_Integrator) = Options(
#     x_reltol = 8eps(),
#     x_suctol = 2eps(),
#     f_abstol = 8eps(),
#     f_reltol = 8eps(),
#     f_suctol = 2eps(),
#     max_iterations = 10_000,
# )
S = 30
nn_pde_basis = NetworkPDEBasis(S,tanh,:Partially)
t_step = 0.1
x_span = (0.0,1.0)
nn_int = NN_PDE_Integrator(nn_pde_basis,RT = 4,RX = 16, xspan = x_span, tstep = t_step,μ =:BSplineDirichlet,λ =:BSplineDirichlet,
                            k_μ = 4,k_λ₀_x = 4,nepochs= 1,initial_guess_method = OGA2D())
lpde = MultiSymplectic.LinearTransport.lpdeproblem(tstep = t_step,tspan =(0.0,t_step),xspan = x_span)

log_file="logs/NN_pde.txt"
open(log_file, "w") do io
    redirect_stdout(io) do
        sol = MultiSymplectic.integrate(lpde,nn_int)
        plot(sol.sol.u[1], label="NN-PDE Solution at t=0.1")
        plot!(lpde.exact_u.(0.1,collect(x_span[1]:0.01:x_span[2])), label="Exact Solution at t=0.1")
        savefig("logs/NN_pde.png")
    end
end


# plot(0.:0.01:1,lpde.exact_u.(0.96,collect(0.:0.01:1)))
# plot!(0.5:0.01:0.6,sol.sol.u[48])

plot(lpde.exact_u.(0.1,collect(0.:0.01:1)), label="Exact Solution at t=0.1")
# plot!(sol.sol.u[3])

plot(sol.sol.u[1])
# NP = parameterlength(u_network)
# sol_params = NeuralNetworkParameters(MultiSymplectic.reconstruct_params(sol.internal.x[10][1:NP], pnn.params))
# u_network([0.3,0.4],sol_params)[1]


# x_vertics = [x_span[1], x_span[2], x_span[2], x_span[1]]
# y_vertics = [-0.1,-0.1, 1.0,1.0]
# x_ls = collect(0.2:0.01:1)


# PNN = NeuralNetwork(NN)
# PNN.params.L1.W[:,2] .= [-1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0]
# PNN.params.L1.W[:,1] .= - 0.2 * [-1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0]
# PNN.params.L1.b[:] = [ 0.9995, -0.0000,  0.6479,  0.3423, -0.5156, -0.7178,  0.2222,  0.7964,
#     -0.4072,  0.4653, -0.2852,  0.5786,  0.1450, -0.3730, -0.4873, -0.6123,
#     -0.2559,  0.4321,  0.3091,  0.3911]
# PNN.params.L2.W[:] = [-5.5742e+00,  1.1024e+01, -5.3139e+00, -4.0719e+00,  6.0699e+01,
#         -7.4047e-01, -9.1054e-01, -1.7195e-01, -1.5121e+02,  4.6744e+01,
#         2.2065e+01,  2.3221e+01,  1.1691e-01, -2.3493e+01,  1.2785e+02,
#         -3.6581e+01, -4.0092e+00, -4.0954e+01, -3.2466e+01,  8.3772e+00]

# function exact_u(t,x)
#     PNN([t,x])[1]
# end


# linear_transport_anim = @animate for (i, tt) in enumerate(0:t_step:0.9)
#     plot(x_ls,exact_u.(tt,x_ls),label="Analytic Solution",size = (900,500), ylims = (-0.1,1.0))
#     sol_params = NeuralNetworkParameters(MultiSymplectic.reconstruct_params(sol.internal.x[i+1][1:NP], pnn.params))

#     plot!(x_ls,[u_network([tt,xx],sol_params)[1] for xx in x_ls],label = "NVI Solution")
#     # plot!(Shape(x_vertics, y_vertics), label = "Computation Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
#     # scatter!(x_span[2] .* QuadratureRules.GaussLegendreQuadrature(RX).nodes,-0.18 * ones(RX),label = "Quadrature Points", color = :red, markersize = 5, markershape = :x)
#     title!("t = $tt")
#     xlabel!("x")
#     ylabel!("u")
# end 
# gif(linear_transport_anim, "figures/linear_transport_eqn2.gif",fps = 5)


# sol_params = NeuralNetworkParameters(MultiSymplectic.reconstruct_params(sol.internal.x[8][1:NP], pnn.params))

# plot!(x_ls,[u_network([0.16,xx],sol_params)[1] for xx in x_ls],label = "NN Solution")
# plot!(x_ls,lpde.exact_u.(0.96,x_ls),label = "Exact Solution")


# x_ls =  collect(0:0.01:1)
# t_ls = collect(0:0.01:1)
# u_values = zeros(length(t_ls), length(x_ls))
# for xx in eachindex(x_ls)
#     for tt in eachindex(t_ls)
#         u_values[tt, xx] = PNN([t_ls[tt], x_ls[xx]])[1]
#     end
# end

# plot(x_ls, t_ls, u_values, xlabel="x", ylabel="t", zlabel="u(x,t)", title="u(x,t) surface")

# test = @animate for (i, t) in enumerate(t_ls)
#     plot(u_values[i, :], x_ls, label="t = $t", xlabel="x", ylabel="u(x,t)", title="u(x,t) at t = $t")
# end 
# gif(test, fps = 5)