struct Trial_Solution_Basis <: AbstractPDEBasis
    activation_function
    NP
    basis_network
    function Trial_Solution_Basis(NN_width, activation;d = 2)
        NN = NeuralNetwork(Chain(Dense(d, NN_width, activation),Dense(NN_width,1,identity,use_bias = false)),initializer = ZeroInitializer())
        new{}(activation, NN_width, NN)
    end
end
