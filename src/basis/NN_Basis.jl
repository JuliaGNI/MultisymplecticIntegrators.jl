struct NN_Basis <: AbstractPDEBasis
    network_arch
    
    u_basis
    v_basis
    w_basis

    u
    NP::Int
    function NN_Basis(u_basis_arch,NP;input_dim=2)
        u_basis = AbstractNeuralNetworks.NeuralNetwork(u_basis_arch)
        # u_grad(x) = Zygote.jacobian(input -> u_func(input),x)[1]
        v_basis(x,ps) = Zygote.jacobian(input -> u_basis(input,ps),x)[1][:,1]
        w_basis(x,ps) = Zygote.jacobian(input -> u_basis(input,ps),x)[1][:,2]

        u_arch = Chain(u_basis_arch...,Dense(NP,1,identity,use_bias=false))
        u_func = AbstractNeuralNetworks.NeuralNetwork(u_arch)
        return new(u_arch,
            u_basis, v_basis, w_basis,
            u_func,
            NP)
    end
end