struct NetworkPDEBasis{OMT} <: AbstractPDEBasis
    network_arch
    
    u
    v
    w    

    ∂u∂P # derivatives with respect to P
    ∂v∂P
    ∂w∂P

    NP::Int
    optim_mode::OMT # :Partially or :Fully, whether to solve only last layer parameters or all the parameters in the neural network
    function NetworkPDEBasis(u_network, optim_mode::OMT = :Partially) where {OMT} 
        sym_u = SymbolicNeuralNetworks.SymbolicNeuralNetwork(u_network)
        u_func = AbstractNeuralNetworks.NeuralNetwork(u_network)
        if optim_mode == :Fully
            NP = AbstractNeuralNetworks.parameterlength(u_network)
        elseif optim_mode == :Partially
            NP = size(u_func.params[keys(u_func.params)[end]].W, 2)
        else
            error("Invalid optim_mode. Use :Partially or :Fully.")
        end

        sym_∂u∂P = SymbolicNeuralNetworks.derivative(SymbolicNeuralNetworks.Gradient(sym_u))[1]
        ∂u∂P_func = SymbolicNeuralNetworks.build_nn_function(sym_∂u∂P, sym_u.params, sym_u.input)

        jac = SymbolicNeuralNetworks.Jacobian(sym_u)
        sym_v = SymbolicNeuralNetworks.derivative(jac)[1]
        sym_w = SymbolicNeuralNetworks.derivative(jac)[2]
        v_func = SymbolicNeuralNetworks.build_nn_function(sym_v, sym_u.params, sym_u.input)
        w_func = SymbolicNeuralNetworks.build_nn_function(sym_w, sym_u.params, sym_u.input)
        
        g = SymbolicNeuralNetworks.Gradient(SymbolicNeuralNetworks.derivative(jac), sym_u)
        sym_∂v∂P = SymbolicNeuralNetworks.derivative(g)[1]
        sym_∂w∂P = SymbolicNeuralNetworks.derivative(g)[2]
        ∂v∂P_func = SymbolicNeuralNetworks.build_nn_function(sym_∂v∂P, sym_u.params, sym_u.input)
        ∂w∂P_func = SymbolicNeuralNetworks.build_nn_function(sym_∂w∂P, sym_u.params, sym_u.input)

        return new{OMT}(u_network,
            u_func, v_func, w_func,
            ∂u∂P_func, ∂v∂P_func, ∂w∂P_func,
            NP,optim_mode)
    end
end