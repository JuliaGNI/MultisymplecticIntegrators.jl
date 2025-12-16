"""
Basis for Lagrangian multipliers at the boundary, for 1D.
"""

struct BSplineDirichlet{T} 
    k::Int # order
    x::AbstractVector{T} # vector to generate knots
    t::AbstractVector{T} # knots from BSplineKit, was optimized

    Bspline
    b #  basis functions 
    function BSplineDirichlet(k::Int) where T
        x = 0:1/(k-1):1
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


# serve as the approximation function inside domain for comparision
struct BSpline2D <: AbstractPDEBasis
    k::Int
    xs 
    ts

    Collocation_x # Collocation matrix after LU factorization
    Collocation_t

    Basis_x
    Basis_t

    Nbasis_x::Int
    Nbasis_t::Int
    S::Int # total number of basis functions
    function BSpline2D(k,;xspan = (0.0,1.0),tstep = 0.05,xstep = 0.02) # tstep and xstep are used to generate the knots, not the same as the problem domain steps
        xs = xspan[1]:xstep:xspan[2]
        ts = 0.0:tstep:1.0

        # Create B-spline knots based on interpolation points (uses an internal function)
        knots_x = SplineInterpolations.make_knots(xs, BSplineOrder(k), nothing)
        knots_t = SplineInterpolations.make_knots(ts, BSplineOrder(k), nothing)

        # Create B-spline bases
        Bx = BSplineBasis(BSplineOrder(k), knots_x; augment = Val(false))
        Bt = BSplineBasis(BSplineOrder(k), knots_t; augment = Val(false))

        # Create and factorise interpolation matrices
        Cx = lu!(collocation_matrix(Bx, xs))
        Ct = lu!(collocation_matrix(Bt, ts))

        Nbasis_x = length(Bx)
        Nbasis_t = length(Bt)
        S = Nbasis_x * Nbasis_t
        return new(k, xs, ts, Cx, Ct, Bx, Bt, Nbasis_x, Nbasis_t, S)
    end

end