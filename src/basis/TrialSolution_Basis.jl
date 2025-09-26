struct Trial_Solution_Basis{TST} <: AbstractPDEBasis
    basis_network
    sol_network
    trial_sol_type::TST  # :TFC or :Plain
    NP
    function Trial_Solution_Basis(basis_network, sol_network, NP,;sol_type::Symbol = :TFC)
        @assert sol_type == :TFC || sol_type == :Plain "sol_type must be :TFC or :Plain"
        new{typeof(sol_type)}(basis_network, sol_network, sol_type, NP)
    end
end
