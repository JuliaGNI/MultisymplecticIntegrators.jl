struct NN_Basis{NT,UT,DT,UFT} <: AbstractPDEBasis
    network_arch::NT
    
    u_basis::UT
    v_basis::DT
    w_basis::DT

    u::UFT
    NP::Int
    function NN_Basis(u_basis_arch,NP;input_dim=2)
        u_basis = AbstractNeuralNetworks.NeuralNetwork(u_basis_arch)
        # u_grad(x) = Zygote.jacobian(input -> u_func(input),x)[1]
        v_basis(x,ps) = Zygote.jacobian(input -> u_basis(input,ps),x)[1][:,1]
        w_basis(x,ps) = Zygote.jacobian(input -> u_basis(input,ps),x)[1][:,2]

        u_arch = Chain(u_basis_arch...,Dense(NP,1,identity,use_bias=false))
        u_func = AbstractNeuralNetworks.NeuralNetwork(u_arch)
        return new{typeof(u_arch),typeof(u_basis),typeof(v_basis),typeof(u_func)}(u_arch,
            u_basis, v_basis, w_basis,
            u_func,
            NP)
    end
end