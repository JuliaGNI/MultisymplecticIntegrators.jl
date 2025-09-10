struct ELM_NN_Basis <: AbstractPDEBasis
    network_arch
    
    u
    v
    w    

    NP::Int
    function ELM_NN_Basis(u_network,NP)
        u_func = AbstractNeuralNetworks.NeuralNetwork(u_network) # how to set the random seed???
        # u_grad(x) = Zygote.jacobian(input -> u_func(input),x)[1]
        v_func(x,ps) = Zygote.jacobian(input -> u_func(input,ps),x)[1][:,1]
        w_func(x,ps) = Zygote.jacobian(input -> u_func(input,ps),x)[1][:,2]
        # maybe using ForwardDiff or Zygote to compute the derivatives, instead of SymbolicNeuralNetworks
        return new(u_network,
            u_func, v_func, w_func,
            NP)
    end
end