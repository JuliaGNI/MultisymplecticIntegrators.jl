struct NetworkPDEBasis <: AbstractPDEBasis
    network_arch
    
    u
    v
    w    

    ∂u∂P # derivatives with respect to P
    ∂v∂P
    ∂w∂P

    NP::Int
    function NetworkPDEBasis(u_network)
        sym_u = SymbolicNeuralNetworks.SymbolicNeuralNetwork(u_network)
        u_func = AbstractNeuralNetworks.NeuralNetwork(u_network)
        NP = AbstractNeuralNetworks.parameterlength(u_network)
        
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

        return new(u_network,
            u_func, v_func, w_func,
            ∂u∂P_func, ∂v∂P_func, ∂w∂P_func,
            NP)
    end
end