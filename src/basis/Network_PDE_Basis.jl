struct NetworkPDEBasis{OMT,AF,UF,VF,WF,UPT,VPT,WPT} <: AbstractPDEBasis
    S::Int
    activation_function::AF

    u::UF
    v::VF
    w::WF

    ∂u∂P::UPT # derivatives with respect to P
    ∂v∂P::VPT
    ∂w∂P::WPT

    NP::Int
    optim_mode::OMT # :Partially or :Fully, whether to solve only last layer parameters or all the parameters in the neural network
    function NetworkPDEBasis(S, activation_function::AF,optim_mode::OMT = :Partially; XT_dim = 2) where {OMT,AF} 
        u_network = Chain(Dense(XT_dim, S, activation_function,),Dense(S, 1,identity,use_bias = false))        
        sym_u = SymbolicNeuralNetworks.SymbolicNeuralNetwork(u_network)
        u_func = AbstractNeuralNetworks.NeuralNetwork(u_network,initializer = ZeroInitializer())
        if optim_mode == :Fully
            NP = AbstractNeuralNetworks.parameterlength(u_network)
        elseif optim_mode == :Partially
            NP = S
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

        return new{OMT,AF,typeof(u_func),typeof(v_func),typeof(w_func),typeof(∂u∂P_func),typeof(∂v∂P_func),typeof(∂w∂P_func)}(S, activation_function,
            u_func, v_func, w_func,
            ∂u∂P_func, ∂v∂P_func, ∂w∂P_func,
            NP,optim_mode)
    end
end
