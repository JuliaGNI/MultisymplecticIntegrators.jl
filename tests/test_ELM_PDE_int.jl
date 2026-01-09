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
elm_int = ELM_PDE_int(nn_elm_basis; RT=32,RX = 18,xspan = xspan,initial_guess_method = LSGD(), show_status=true)
lpde = MultiSymplectic.Wave.lpdeproblem(timestep=h, timespan =(0.0,h),xspan = xspan)

log_file="logs/elmint.txt"
open(log_file, "w") do io
    redirect_stdio(stdout=log_file, stderr=log_file) do
        sol = MultiSymplectic.integrate(lpde,elm_int)
    end
end