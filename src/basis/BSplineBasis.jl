"""
Basis for Lagrangian multipliers at the boundary, for 1D.
"""

struct BSplineDirichlet{T} 
    k::Int # order
    x::AbstractVector{T} # vector to generate knots
    t::AbstractVector{T} # knots from BSplineKit, was optimized

    Bspline
    b #  basis functions 
    function BSplineDirichlet(k::Int,x::AbstractVector{T}) where T
        t = make_knots(x,BSplineOrder(k),nothing) # nothing is the bc,default is Dirichlet
        B = BSplineBasis(BSplineOrder(k), t;augment = Val(false))
        basis_fct = []
        for i in eachindex(B)
            push!(basis_fct, B[i])
        end

        return new{T}(k, x, t, B, basis_fct)
    end
end

Base.length(Basis::BSplineDirichlet) = Base.length(Basis.b)
