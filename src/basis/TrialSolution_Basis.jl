struct Trial_Solution_Basis <: AbstractPDEBasis
    activation_function
    S
    NP
    # basis_network
    # sol_network
    function Trial_Solution_Basis(NN_width, activation;d = 2)
        # BNN = NeuralNetwork(Chain(Dense(d, NN_width, activation)),initializer = ZeroInitializer())
        # PNN = NeuralNetwork(Chain(Dense(d, NN_width, activation),Dense(NN_width,1,identity,use_bias = false)),initializer = ZeroInitializer())
        new{}(activation, NN_width, 4 * NN_width)#, BNN, PNN
    end
end

