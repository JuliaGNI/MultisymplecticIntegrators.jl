struct Trial_Solution_Basis{TST} <: AbstractPDEBasis
    PNN
    trial_sol_type::TST  # :TFC or :Plain
    NP
    function Trial_Solution_Basis(PNN,NP,;sol_type::Symbol = :TFC)
        @assert sol_type == :TFC || sol_type == :Plain "sol_type must be :TFC or :Plain"
        new{typeof(sol_type)}(PNN, sol_type, NP)
    end
end
