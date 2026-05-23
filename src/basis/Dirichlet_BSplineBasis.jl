"""
Basis for Lagrangian multipliers at the boundary, for 1D.
"""
# serve as the approximation function inside domain
struct Dirichlet_BSpline2D{CXT, CTT, BXT, BTT} <: AbstractPDEBasis
    k::Int
    xspan::Tuple{Float64,Float64}
    timestep::Float64

    x_knot_interval::Float64
    xs::Vector{Float64} 

    t_knot_interval::Float64
    ts::Vector{Float64} 

    collocation_points_x::Vector{Float64}
    collocation_points_t::Vector{Float64}

    collocation_matrix_x::CXT # Collocation matrix after LU factorization
    collocation_matrix_t::CTT

    Basis_x::BXT
    Basis_t::BTT

    Nbasis_x::Int
    Nbasis_t::Int
    S::Int # total number of basis functions
    function Dirichlet_BSpline2D(k;timestep = 0.1, xspan = (0.0,1.0), t_knot_interval = 0.1, x_knot_interval = 0.05) # tstep and xstep are used to generate the knots, not the same as the problem domain steps
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
        return new{typeof(Cx), typeof(Ct), typeof(Rx), typeof(Bt)}(k,xspan,timestep,x_knot_interval, xs, t_knot_interval,ts, x_collocation_points, t_collocation_points, Cx, Ct, Rx, Bt, Nbasis_x, Nbasis_t, S)
    end

end

function Base.show(io::IO, basis::Dirichlet_BSpline2D)
    print(io, "\n 1+1 Tensor Product B-spline Basis, Only for Zero Dirichlet Boundary Condition with:\n")
    print(io, "   Order in each dimension k:$(basis.k) \n")
    print(io, "   Nbasis_x: $(basis.Nbasis_x), Nbasis_t: $(basis.Nbasis_t), total DOFs:$(basis.S) \n")
    print(io, "   Break Length x: $(basis.x_knot_interval), Break Length t:$(basis.t_knot_interval) \n")
    print(io, "   Scaled Breaks_x: $(basis.xs) \n")
    print(io, "   Scaled Breaks_t: $(basis.ts) \n")
end