struct ELM_NN_Basis <: AbstractPDEBasis
    network_arch
    
    u
    v
    w    

    NP::Int
    function ELM_NN_Basis(u_network,NP)
        sym_u = SymbolicNeuralNetworks.SymbolicNeuralNetwork(u_network)
        u_func = AbstractNeuralNetworks.NeuralNetwork(u_network)
        jac = SymbolicNeuralNetworks.Jacobian(sym_u)
        sym_v = SymbolicNeuralNetworks.derivative(jac)[:,1]
        sym_w = SymbolicNeuralNetworks.derivative(jac)[:,2]
        v_func = SymbolicNeuralNetworks.build_nn_function(sym_v, sym_u.params, sym_u.input)
        w_func = SymbolicNeuralNetworks.build_nn_function(sym_w, sym_u.params, sym_u.input)
        # maybe using ForwardDiff or Zygote to compute the derivatives, instead of SymbolicNeuralNetworks
        return new(u_network,
            u_func, v_func, w_func,
            NP)
    end
end