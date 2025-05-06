"""
Basis for Lagrangian multipliers at the boundary, for 1D.
"""

struct BSplineDirichlet{T} 
    k::Int # order
    t::AbstractVector{T} # vector to generate knots

    knot_seq
    b #  basis functions 
    function BSplineDirichlet(k::Int,t::AbstractVector{T}) where T
        B = BSplineBasis(BSplineOrder(k), t)
        knot_seq = B.t
        basis_fct = []
        for i in eachindex(B)
            push!(basis_fct, B[i])
        end

        return new{T}(k, t, knot_seq, basis_fct)
    end
end

Base.length(Basis::BSplineDirichlet) = Base.length(Basis.b)
