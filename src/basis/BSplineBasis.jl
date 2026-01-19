"""
Basis for Lagrangian multipliers at the boundary, for 1D.
"""

struct BSplineDirichlet 
    k::Int # order
    Nbasis::Int # number of basis
    x::AbstractVector # vector to generate knots
    t::AbstractVector # knots from BSplineKit, was optimized

    b #  basis functions 
    function BSplineDirichlet(Nbasis::Int,k,a,b)
        QGau = QuadratureRules.GaussLegendreQuadrature(Nbasis)
        x = a:(b - a)/(Nbasis-1):b
        # x = a .+ (b-a) .* QGau.nodes
        t = make_knots(x,BSplineOrder(k),nothing) # nothing is the bc,default option is Dirichlet
        B = BSplineBasis(BSplineOrder(k), t;augment = Val(false))

        @assert Nbasis == length(B) "Number of basis functions does not match Nbasis"
        return new(k, Nbasis, x, t, B)
    end
end

Base.length(Basis::BSplineDirichlet) = Basis.Nbasis


# serve as the approximation function inside domain
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
    function BSpline2D(k,;xspan = (0.0,1.0),t_knot_interval = 0.1,x_knot_interval = 0.05) # tstep and xstep are used to generate the knots, not the same as the problem domain steps
        xs = collect(xspan[1]:x_knot_interval:xspan[2])
        ts = collect(0.0:t_knot_interval:1.0)

        # Create B-spline knots based on interpolation points 
        # The knots are not the same with the breaks,ie the xs and ts
        # No boundary condition is imposed here
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
