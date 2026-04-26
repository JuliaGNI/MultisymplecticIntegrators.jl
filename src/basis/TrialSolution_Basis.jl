struct Trial_Solution_Basis{AF} <: AbstractPDEBasis
    activation_function::AF
    S::Int
    NP::Int
    # basis_network
    # sol_network
    function Trial_Solution_Basis(NN_width::Int, activation::AF;d = 2) where {AF}
        # BNN = NeuralNetwork(Chain(Dense(d, NN_width, activation)),initializer = ZeroInitializer())
        # PNN = NeuralNetwork(Chain(Dense(d, NN_width, activation),Dense(NN_width,1,identity,use_bias = false)),initializer = ZeroInitializer())
        new{AF}(activation, NN_width, 4 * NN_width)#, BNN, PNN
    end
end

