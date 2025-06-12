using GeometricIntegrators
using QuadratureRules
using MultiSymplectic
using AbstractNeuralNetworks
using Random

# by default, the first dimension is time, the second is space, 
# and just consider 1+1 now. 
u_network = Chain(
    Dense(2, 10, tanh),
    Dense(10, 10, tanh),
    Dense(10, 1,identity,use_bias = false)
)

# pnn = NeuralNetwork(u_network)
# nn_pde_basis = NetworkPDEBasis(u_network)
# t_step = 0.05
# x_span = (0.,0.5)
# nn_int = NN_PDE_Integrator(nn_pde_basis,RT = 4,RX = 8, xspan = x_span, tstep = t_step,μ =:BSplineDirichlet,λ =:BSplineDirichlet,k_μ = 4,k_λ₀_x = 4)

# lpde = MultiSymplectic.SineGordon.lpdeproblem(tstep = t_step,tspan =(0.0,5.5),xspan = x_span)
# sol = MultiSymplectic.integrate(lpde,nn_int)


nn = NeuralNetwork(u_network)
function reinit_params_with_boxinit!(params::NeuralNetworkParameters)
    layer_names = keys(params)
    layer_values = values(params)  # <- use values() instead of getfield

    new_layers = NamedTuple()

    for (name, layer) in zip(layer_names, layer_values)
        in_size = size(layer.W, 2)
        out_size = size(layer.W, 1)
        if hasfield(typeof(layer), :b)
            layer.W, layer.b = MultiSymplectic.box_init_plain(in_size, out_size)
        else
            # For layers without bias (e.g., output), just regenerate W
            layer.W, _ = box_init_plain(in_size, out_size)

        end
    end

    return NeuralNetworkParameters(new_layers)
end


reinit_params_with_boxinit!(nn.params)