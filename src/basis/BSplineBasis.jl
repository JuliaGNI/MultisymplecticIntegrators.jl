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


# serve as the approximation function inside domain for comparision
struct BSpline2D{T}
    k::Int
    xs 
    ts

    Collocation_x
    Collocation_t
    function BSpline2D(k,tspan,xspan,;tstep = 0.01,xstep = 0.1)
        xs = xspan[1]:xstep:xspan[2]
        ts = tspan[1]:tstep:tspan[2]

        # Create B-spline knots based on interpolation points (uses an internal function)
        knots_x = SplineInterpolations.make_knots(xs, k, nothing)
        knots_t = SplineInterpolations.make_knots(ts, k, nothing)

        # Create B-spline bases
        Bx = BSplineBasis(k, knots_x; augment = Val(false))
        Bt = BSplineBasis(k, knots_t; augment = Val(false))

        # Create and factorise interpolation matrices
        Cx = lu!(collocation_matrix(Bx, xs))
        Ct = lu!(collocation_matrix(Bt, ts))
        return new{T}(k, xs, ts, Cx, Ct)
    end

end