using QuadratureRules
using MultiSymplectic
using AbstractNeuralNetworks
using Random
using GeometricMachineLearning
using Zygote
using Plots
using ForwardDiff
using Infiltrator
using GeometricIntegratorsBase

GeometricIntegratorsBase.default_options(::ELM_PDE_int) = (
    x_abstol = 8eps(),
    f_abstol = 8eps(),
    max_iterations = 100,
)

Random.seed!(123)
relu3(x) = max.(0, x).^3
activation = tanh
S = 200
u_basis = Chain(
    Dense(2, S, activation),
    Dense(S, S, activation),
    Dense(S, S, activation),
)   

nn_elm_basis = NN_Basis(u_basis, S)

xspan = (0.3,0.8)
h = 0.3
elm_int = ELM_PDE_int(nn_elm_basis; RT=32,RX = 18,xspan = xspan,initial_guess_method = LSGD(), show_status=false)
lpde = MultiSymplectic.Wave.lpdeproblem(timestep=h, timespan =(0.0,h),xspan = xspan)

log_file="logs/elmint.txt"
open(log_file, "w") do io
    redirect_stdio(stdout=log_file, stderr=log_file) do
        sol = MultiSymplectic.integrate(lpde,elm_int)
        
        p = @layout [a b c]
        p1 = plot([lpde.exact_u(h,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_u")
        plot!(p1, sol.u[1], label="sol.u")
        p2 = plot([lpde.exact_v(h,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_v")
        plot!(p2, sol.v[1], label="sol.v")
        p3 = plot([lpde.exact_w(h,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_w")
        plot!(p3, sol.w[1], label="sol.w")
        p = plot(p1, p2, p3, layout=p)
        savefig("logs/ELMInt_t=h.pdf")
    end
end