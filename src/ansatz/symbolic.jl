"""
    SymbolicAnsatz(expr, t, x, p; xspan, start, degree = 1, ncells = (2, 2), n = nothing)

A symbolic ansatz of one space-time slab whose field is a closed-form expression.

The slab is the rectangle `[0, 1] × xspan` in the normalised time coordinate `t̂` and the physical
space coordinate `x`, and the field is

```math
u(t̂, x) = \\mathrm{expr}(t̂, x, p),
```

where `expr` is a `Num` or a `Vector{Num}` of `D` entries, `p` the vector of its parameter
symbols, and `start` the initial guess of the parameter vector on the first slab. The field's time
is the slab's normalised time, so an expression in physical time is substituted before it is given
here.

The constructor builds six in-place functions with `Symbolics.build_function(…; expression =
Val(false))`: the field, its two derivatives `∂u/∂t̂` and `∂u/∂x`, and the parameter Jacobian of
each, a `D × length(p)` matrix. They are held in concrete fields, so no container of abstract type
and no `eval` of generated code enters a step; each function compiles once, at its first call. The
ansatz also holds a [`TensorBSplineAnsatz`](@ref) named `trace`, whose rule is the slab's
quadrature and whose space basis carries the coupling's multiplier tables; the trace basis does not
constrain the field.

The constructor refuses an expression whose free symbols are not `t`, `x` and the entries of `p`,
an empty `p`, a `start` of another length than `p` and an `xspan` that does not increase, each with
an `ArgumentError` naming the offending value. [`check_ansatz`](@ref) refuses an expression count
that is not the problem's `D`, an element type that is not the problem's, and an `xspan` that is
not the problem's.

The parameters are the unknowns of the slab. [`WeakInitialAndBoundaryConditions`](@ref) is the one
coupling the ansatz takes, and it lets the field be free at both ends of `xspan`, so the trace
basis is built with the `Free()` boundary condition that coupling needs.

The integrator stores the solved top multipliers as the `top_multiplier` entry of each step's
internal state, as it does for a spline ansatz. For a symbolic ansatz they are not unique: the
Jacobian of a step has a null space in the top and side multipliers that leaves the parameters
determined, so the stored values carry no meaning on their own.

```julia
SymbolicAnsatz(p[1] * sin(p[2] * x - p[3] * t), t, x, p;
    xspan = (0, 2π), start = [1, 1, 2π])
```
"""
struct SymbolicAnsatz{T, TB <: TensorBSplineAnsatz, U, UT, UX, JU, JUT, JUX} <:
       SpaceTimeAnsatz
    expr::Vector{Num}
    t::Num
    x::Num
    p::Vector{Num}
    start::Vector{T}
    trace::TB
    xspan::NTuple{2, T}
    num_params::Int
    u::U
    ut::UT
    ux::UX
    Ju::JU
    Jut::JUT
    Jux::JUX
end

function SymbolicAnsatz(expr, t::Num, x::Num, p::AbstractVector;
        xspan, start::AbstractVector, degree = 1, ncells = (2, 2), n = nothing)
    expressions = expr isa AbstractVector ? collect(expr) : [expr]
    all(e -> e isa Num, expressions) ||
        throw(ArgumentError("the expression must be a Num or a vector of Num, got " *
                            "$(typeof(expr))"))
    isempty(p) &&
        throw(ArgumentError("the parameter vector is empty; a SymbolicAnsatz needs at least " *
                            "one parameter symbol"))
    num_params = length(p)
    allowed = Num[t, x, p...]
    for expression in expressions
        for variable in Symbolics.get_variables(expression)
            any(isequal(variable), allowed) ||
                throw(ArgumentError("the expression holds the free symbol $(variable), which " *
                                    "is not t, x or an entry of p"))
        end
    end
    length(start) == num_params ||
        throw(ArgumentError("the start vector holds $(length(start)) entries, but the ansatz " *
                            "has $num_params parameters"))
    T = _coordinate_type(xspan)
    xsp = (convert(T, xspan[1]), convert(T, xspan[2]))
    trace = TensorBSplineAnsatz(; degree, ncells, xspan = xsp, n, x_boundary = Free())
    ut = Num[Symbolics.derivative(expression, t) for expression in expressions]
    ux = Num[Symbolics.derivative(expression, x) for expression in expressions]
    u = build_in_place(expressions, p, t, x)
    u_t = build_in_place(ut, p, t, x)
    u_x = build_in_place(ux, p, t, x)
    Ju = build_in_place(parameter_jacobian(expressions, p), p, t, x)
    Jut = build_in_place(parameter_jacobian(ut, p), p, t, x)
    Jux = build_in_place(parameter_jacobian(ux, p), p, t, x)
    return SymbolicAnsatz{T, typeof(trace), typeof(u), typeof(u_t), typeof(u_x),
        typeof(Ju), typeof(Jut), typeof(Jux)}(
        expressions, t, x, Vector{Num}(p), Vector{T}(start), trace, xsp, num_params,
        u, u_t, u_x, Ju, Jut, Jux)
end

# The in-place function of an array expression, which writes a `D`-vector or a `D × length(p)`
# matrix of the arguments' element type.
function build_in_place(expression, p, t, x)
    Symbolics.build_function(expression, p, t, x; expression = Val(false))[2]
end

# The parameter Jacobian of every component, a `D × length(p)` matrix of expressions.
function parameter_jacobian(expressions::Vector{Num}, p::AbstractVector)
    Num[Symbolics.derivative(expressions[d], p[j])
        for d in eachindex(expressions), j in eachindex(p)]
end

"""The space interval of the slab."""
xspan(ansatz::SymbolicAnsatz) = ansatz.xspan

"""The tensor Gauss–Legendre quadrature of the slab, which the ansatz's trace basis owns."""
quadrature(ansatz::SymbolicAnsatz) = quadrature(ansatz.trace)

"""
    check_ansatz(problem, ansatz)

Reject a symbolic ansatz that the problem cannot support, when the integrator is built.

The expression count must be the problem's `D`, the element type the problem's, and the `xspan`
the problem's. Each refusal is an `ArgumentError` that names the two values it compared, and the
element type is compared first, so a pair that mismatches in both hears about the type.
"""
function check_ansatz(problem::LagrangianPDEProblem{PT}, ansatz::SymbolicAnsatz{T}) where {
        PT, T}
    T === PT || throw(ArgumentError(
        "the ansatz element type $T does not match the problem's $PT"))
    D = components(problem)
    length(ansatz.expr) == D ||
        throw(ArgumentError("the ansatz holds $(length(ansatz.expr)) expressions, but the " *
                            "problem has $D components"))
    xspan(ansatz) == xspan(problem) || throw(ArgumentError(
        "the ansatz xspan $(xspan(ansatz)) does not match the problem's $(xspan(problem))"))
    return nothing
end

"""
    SymbolicSlab{ST, T, A}

The block of a [`SymbolicAnsatz`](@ref) in the slab cache: what the ansatz keeps of one slab.

`params` is the parameter vector of the slab and `residual` its interior weak form, of one row per
parameter. `u`, `v` and `w` hold the field and its derivatives at the nodes of the slab's
quadrature, and `Ju`, `Jv` and `Jw` the parameter Jacobian of each there. `bottom`, `top`, `left`
and `right` hold the field and its parameter Jacobian at the four boundary sets: `t̂ = 0` and
`t̂ = 1` at the space nodes, and `x = a` and `x = b` at the time nodes. `tabulation` holds the
tables of the slab's quadrature, and `output` and `trace` evaluate the trace basis' space basis at
the output grid and at the nodes of the space rule. The six evaluators of the ansatz write into
the scratch buffers `ubuf` to `Juxbuf`, so a call of `slab_field!`, `slab_weak_form!` or
`top_values!` allocates nothing; `lambda_bottom`, `lambda_top`, `mu_left` and `mu_right` hold the
multipliers' values at the nodes of the rule.
"""
struct SymbolicSlab{ST, T, A <: SymbolicAnsatz{T}}
    ansatz::A
    params::Vector{ST}
    residual::Vector{ST}
    u::Array{ST, 3}
    v::Array{ST, 3}
    w::Array{ST, 3}
    Ju::Array{ST, 4}
    Jv::Array{ST, 4}
    Jw::Array{ST, 4}
    bottom_u::Matrix{ST}
    bottom_J::Array{ST, 3}
    top_u::Matrix{ST}
    top_J::Array{ST, 3}
    left_u::Matrix{ST}
    left_J::Array{ST, 3}
    right_u::Matrix{ST}
    right_J::Array{ST, 3}
    tabulation::SlabTabulation{T}
    output::PointEvaluation{T}
    trace::PointEvaluation{T}
    ubuf::Vector{ST}
    utbuf::Vector{ST}
    uxbuf::Vector{ST}
    Jubuf::Matrix{ST}
    Jutbuf::Matrix{ST}
    Juxbuf::Matrix{ST}
    lambda_bottom::Matrix{ST}
    lambda_top::Matrix{ST}
    mu_left::Matrix{ST}
    mu_right::Matrix{ST}
end

"""
    ansatz_data(ansatz, problem, ST)

The ansatz's block of the slab cache, built once per integrator.
"""
function ansatz_data(a::SymbolicAnsatz{T}, problem::LagrangianPDEProblem,
        ::Type{ST}) where {T, ST}
    D = length(a.expr)
    np = a.num_params
    tabulation = SlabTabulation(quadrature(a.trace))
    xb = space_basis(a.trace)
    qt = length(tabulation.time.node)
    qx = length(tabulation.space.node)
    return SymbolicSlab{ST, T, typeof(a)}(a,
        zeros(ST, np), zeros(ST, np),
        zeros(ST, qt, qx, D), zeros(ST, qt, qx, D), zeros(ST, qt, qx, D),
        zeros(ST, qt, qx, D, np), zeros(ST, qt, qx, D, np), zeros(ST, qt, qx, D, np),
        zeros(ST, qx, D), zeros(ST, qx, D, np), zeros(ST, qx, D), zeros(ST, qx, D, np),
        zeros(ST, qt, D), zeros(ST, qt, D, np), zeros(ST, qt, D), zeros(ST, qt, D, np),
        tabulation, PointEvaluation(xb, xgrid(problem)),
        PointEvaluation(xb, tabulation.space.node),
        zeros(ST, D), zeros(ST, D), zeros(ST, D),
        zeros(ST, D, np), zeros(ST, D, np), zeros(ST, D, np),
        zeros(ST, qx, D), zeros(ST, qx, D), zeros(ST, qt, D), zeros(ST, qt, D))
end

"""
    ansatz_data_type(ansatz, problem, ST)

The type of [`ansatz_data`](@ref)`(ansatz, problem, ST)`, computed without building the block.
"""
function ansatz_data_type(a::SymbolicAnsatz{T}, ::LagrangianPDEProblem,
        ::Type{ST}) where {T, ST}
    return SymbolicSlab{ST, T, typeof(a)}
end

"""
    evaluate_at!(slab, t̂, x)

Evaluate the six functions of the ansatz at one point into the block's scratch: the field, its
two derivatives and their three parameter Jacobians.
"""
@inline function evaluate_at!(s::SymbolicSlab{ST}, t̂, x) where {ST}
    a = s.ansatz
    a.u(s.ubuf, s.params, t̂, x)
    a.ut(s.utbuf, s.params, t̂, x)
    a.ux(s.uxbuf, s.params, t̂, x)
    a.Ju(s.Jubuf, s.params, t̂, x)
    a.Jut(s.Jutbuf, s.params, t̂, x)
    a.Jux(s.Juxbuf, s.params, t̂, x)
    return nothing
end

"""The normalised time nodes and the space nodes of the slab's quadrature."""
function quadrature_points(s::SymbolicSlab)
    return (s.tabulation.time.node, s.tabulation.space.node)
end

"""The evaluation of the trace basis at the output grid."""
output_evaluation(s::SymbolicSlab) = s.output

"""The evaluation of the trace basis at the nodes of the space quadrature."""
trace_evaluation(s::SymbolicSlab) = s.trace

# The field and its parameter Jacobian at the points `(t̂, x)` of one edge of the slab, one row
# per point: the space nodes at `t̂ = 0` and `t̂ = 1`, and the time nodes at either end of `xspan`.
function edge!(u::AbstractMatrix{ST}, J::AbstractArray{ST, 3}, s::SymbolicSlab{ST},
        points) where {ST}
    for (k, (t̂, x)) in enumerate(points)
        evaluate_at!(s, t̂, x)
        for d in axes(u, 2)
            u[k, d] = s.ubuf[d]
            for i in axes(J, 3)
                J[k, d, i] = s.Jubuf[d, i]
            end
        end
    end
    return nothing
end

"""
    slab_field!(u, v, w, slab, x, bottom, nf, h)

The field of the slab that the solver's vector `x` describes, at the nodes of the slab's
quadrature and at the four boundary sets: the parameter vector from the first `nf` entries of
`x`, and the field, its derivatives and their parameter Jacobians from it.

`v` is the derivative in physical time, `∂_{t̂}u / h`, and `w` the derivative in space; the
parameter Jacobian of `v` carries the same `1/h`. A symbolic ansatz fixes no bottom row, so
`bottom` is `nothing`.
"""
function slab_field!(
        u::AbstractArray{ST, 3}, v::AbstractArray{ST, 3}, w::AbstractArray{ST, 3},
        s::SymbolicSlab{ST}, x::AbstractVector{ST}, bottom, nf::Integer,
        h) where {ST}
    a = s.ansatz
    np = a.num_params
    D = length(a.expr)
    for i in 1:np
        s.params[i] = x[i]
    end
    hh = convert(ST, h)
    tq, xq = s.tabulation.time, s.tabulation.space
    qt, qx = nquadrature(tq), nquadrature(xq)
    for q in 1:qt, r in 1:qx

        evaluate_at!(s, tq.node[q], xq.node[r])
        for d in 1:D
            u[q, r, d] = s.ubuf[d]
            v[q, r, d] = s.utbuf[d] / hh
            w[q, r, d] = s.uxbuf[d]
            for i in 1:np
                s.Ju[q, r, d, i] = s.Jubuf[d, i]
                s.Jv[q, r, d, i] = s.Jutbuf[d, i] / hh
                s.Jw[q, r, d, i] = s.Juxbuf[d, i]
            end
        end
    end
    left, right = convert(ST, xspan(a)[1]), convert(ST, xspan(a)[2])
    edge!(s.bottom_u, s.bottom_J, s, zip(Iterators.repeated(zero(ST)), xq.node))
    edge!(s.top_u, s.top_J, s, zip(Iterators.repeated(one(ST)), xq.node))
    edge!(s.left_u, s.left_J, s, zip(tq.node, Iterators.repeated(left)))
    edge!(s.right_u, s.right_J, s, zip(tq.node, Iterators.repeated(right)))
    return nothing
end

"""The interior weak form of the slab, from the density's derivatives at its nodes."""
function slab_weak_form!(s::SymbolicSlab{ST}, ∂u, ∂v, ∂w, h) where {ST}
    residual = s.residual
    fill!(residual, zero(ST))
    tq, xq = s.tabulation.time, s.tabulation.space
    D = size(s.u, 3)
    hh = convert(ST, h)
    for i in eachindex(residual)
        total = zero(ST)
        for q in 1:nquadrature(tq), r in 1:nquadrature(xq)

            ω = tq.weight[q] * xq.weight[r]
            for d in 1:D
                total += ω * (∂u[q, r, d] * s.Ju[q, r, d, i] +
                          ∂v[q, r, d] * s.Jv[q, r, d, i] +
                          ∂w[q, r, d] * s.Jw[q, r, d, i])
            end
        end
        residual[i] = hh * total
    end
    return nothing
end

"""
    free_residual!(b, slab, nf)

The interior weak form, into the first `nf` entries of the solver's vector.
"""
function free_residual!(b::AbstractVector, s::SymbolicSlab, nf::Integer)
    for i in 1:nf
        b[i] = s.residual[i]
    end
    return nothing
end

"""A symbolic ansatz carries no top row of coefficients: the guess continues its parameter vector."""
keep_top!(::SymbolicSlab) = nothing

"""
    top_values!(u, v, w, slab, evaluation, h)

The field and its two derivatives at the top of the slab, `t̂ = 1`, at the points of `evaluation`,
into `u[i, d]`, `v[i, d]` and `w[i, d]`. `∂ₜu` is the derivative in the normalised time divided by
the slab's `h`.
"""
function top_values!(u::AbstractMatrix{ST}, v::AbstractMatrix{ST}, w::AbstractMatrix{ST},
        s::SymbolicSlab{ST}, evaluation::PointEvaluation, h) where {ST}
    hh = convert(ST, h)
    D = size(u, 2)
    fill!(u, zero(ST))
    fill!(v, zero(ST))
    fill!(w, zero(ST))
    for i in eachindex(evaluation.points)
        evaluate_at!(s, one(ST), convert(ST, evaluation.points[i]))
        for d in 1:D
            u[i, d] = s.ubuf[d]
            v[i, d] = s.utbuf[d] / hh
            w[i, d] = s.uxbuf[d]
        end
    end
    return nothing
end

"""
    total_free_coefficients(method, problem)

The size of one symbolic slab's nonlinear system: its parameters, and the coupling's unknowns
after them.
"""
function total_free_coefficients(method::SpaceTimeGalerkin{A, C, G},
        problem::LagrangianPDEProblem) where {A <: SymbolicAnsatz, C <: SlabCoupling,
        G <: SlabInitialGuess}
    a = ansatz(method)
    return a.num_params + coupling_unknowns(coupling(method), a, problem)
end

"""
    initsolver(solvermethod, method, caches)

The symbolic family solves its slab with a rank-revealing linear solve.

The multiplier coupling's Jacobian is a saddle point whose constraint rows depend only on the
field, so its rank is at most twice the parameter count: for the three-parameter travelling wave
`p₁ sin(p₂x − p₃t̂)` on a trace basis of degree one and two cells, the first slab's `10 × 10`
matrix has rank 6, and its four smallest singular values are at round-off. The minimum-norm step
of a complete orthogonal factorization solves a consistent system exactly and drops the
directions the rank does not carry, where a factorisation without rank detection loses the
descent direction: started at its exact parameters, that run takes three to 29 iterations per
slab with the ecosystem's LU and reaches the tolerance on its first slab alone in `Float64`,
where the pivoted QR converges in two iterations at every slab in both precisions.
"""
function initsolver(solvermethod::Newton,
        method::SpaceTimeGalerkin{<:SymbolicAnsatz}, caches::CacheDict; kwargs...)
    options = merge((; linear_solver_method = SimpleSolvers.LapackPivotedQR()),
        values(kwargs))
    return invoke(initsolver,
        Tuple{NonlinearSolverMethod, GeometricMethod, CacheDict},
        solvermethod, method, caches; options...)
end
