struct Trial_Solution_Basis <: AbstractPDEBasis
    basis_network
    NP
    function Trial_Solution_Basis(basis_network, NP)
        new{}(basis_network, NP)
    end
end
