"""
Basis for Lagrangian multipliers at the boundary, for 1D.
"""
# serve as the approximation function inside domain
struct Dirichlet_BSpline2D{CXT, BXT, BTT} <: AbstractPDEBasis
    k::Int
    xs::Vector{Float64} 
    ts::Vector{Float64} 

    collocation_points_x::Vector{Float64}
    collocation_points_t::Vector{Float64}

    collocation_matrix_x::CXT # Collocation matrix after LU factorization
    collocation_matrix_t::CXT

    Basis_x::BXT
    Basis_t::BTT

    Nbasis_x::Int
    Nbasis_t::Int
    S::Int # total number of basis functions
    function Dirichlet_BSpline2D(k,;xspan = (0.0,1.0),t_knot_interval = 0.1,x_knot_interval = 0.05) # tstep and xstep are used to generate the knots, not the same as the problem domain steps
        xs = collect(xspan[1]:x_knot_interval:xspan[2])
        ts = collect(0.0:t_knot_interval:1.0)

        # Create B-spline bases
        Bx = BSplineBasis(BSplineOrder(k), xs)
        Rx = RecombinedBSplineBasis(Bx, BSplineKit.Derivative(0))

        Bt = BSplineBasis(BSplineOrder(k), ts)

        x_collocation_points = collocation_points(Rx)
        t_collocation_points = collocation_points(Bt)

        # Create and factorise interpolation matrices
        Cx = lu!(collocation_matrix(Rx, x_collocation_points))
        Ct = lu!(collocation_matrix(Bt, t_collocation_points))

        Nbasis_x = length(Rx)
        Nbasis_t = length(Bt)
        S = Nbasis_x * Nbasis_t
        return new{typeof(Cx), typeof(Rx), typeof(Bt)}(k, xs, ts, x_collocation_points, t_collocation_points, Cx, Ct, Rx, Bt, Nbasis_x, Nbasis_t, S)
    end

end
