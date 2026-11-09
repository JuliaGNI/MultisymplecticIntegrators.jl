"""
    NetworkAnsatz(; S, activation = tanh, mode = :all, xspan, ncells, degree,
                    quadrature_order = nothing, D = 1)

A neural-network ansatz of one space-time slab: a two-layer network whose input is the normalised
time `t̂` and the physical space coordinate `x`, and whose output is the `D` components of the field.
`NetworkAnsatz <: SpaceTimeAnsatz`, so a [`SpaceTimeGalerkin`](@ref) holds it as its ansatz.

The network is

```math
u_d(\\hat t, x) = \\sum_{s=1}^{S} W^{(2)}_{ds} \\, \\sigma\\bigl(W^{(1)}_{s1} \\hat t
                 + W^{(1)}_{s2} x + b_s\\bigr),
```

that is `Chain(Dense(2, S, activation), Dense(S, D, identity; use_bias = false))`. Its parameters are
the hidden weights `(S, 2)`, the hidden biases `(S,)` and the output weights `(D, S)`, in the order
`L1.W`, `L1.b`, `L2.W` of `NeuralNetworkParameters`, and one hidden layer serves all `D` components.

`mode` fixes which parameters the solver determines: `:all` frees every one of them, and
`:last_layer` frees the output weights alone and leaves the hidden layer where the initial guess put
it, so that the step is affine in its unknowns, and two Newton iterations reach its tolerance
wherever the output weights carry every constraint row of the coupling. The parameters live in the
ansatz's block in the slab cache and never in the ansatz, so two integrators that share one ansatz
object give identical results.

The ansatz carries a [`TensorBSplineAnsatz`](@ref) as its trace basis, `trace_basis(ansatz)`, built
from `degree` and `ncells` on the same `xspan` with `Free()` at both ends, the condition the weak
initial-and-boundary coupling needs. The functions of that basis are the multipliers of a coupling
that imposes the slab's data weakly, and its rule is the one the coupling's tables and its surface
rows are built on. The field itself is evaluated on a rule of the ansatz's own: `quadrature_order`
Gauss nodes per cell in the time direction, and the space grid of the trace basis, which the cache's
trace is kept on. `nothing` takes one node per cell more than the
trace basis has, so that the two rules are not the same one.

An `xspan` that does not hold two values, an `S` or a `D` that is not positive, and a `mode` that is
neither `:all` nor `:last_layer` raise `ArgumentError`. An activation that the symbolic derivatives
of `SymbolicNeuralNetworks` cannot reach raises that package's own error, which is not caught here.

```julia
ansatz = NetworkAnsatz(S = 10, xspan = (0.0, 1.0), ncells = (2, 5), degree = 3)
```
"""
struct NetworkAnsatz{T, C, FT, JT, TB <: TensorBSplineAnsatz{T}, Q} <: SpaceTimeAnsatz
    chain::C
    xspan::NTuple{2, T}
    mode::Symbol
    trace::TB
    quadrature::Q
    field::FT
    gradient::JT
    S::Int
    D::Int
    np::Int
    nfree::Int
    first_free::Int
end

const NETWORK_MODES = (:all, :last_layer)

function NetworkAnsatz(; S::Integer, activation = tanh, mode::Symbol = :all, xspan,
        ncells, degree, quadrature_order = nothing, D::Integer = 1)
    length(xspan) == 2 ||
        throw(ArgumentError("xspan must hold two values, got $xspan"))
    S ≥ 1 || throw(ArgumentError("S must be at least 1, got $S"))
    D ≥ 1 || throw(ArgumentError("D must be at least 1, got $D"))
    mode in NETWORK_MODES ||
        throw(ArgumentError("mode must be one of $NETWORK_MODES, got :$(mode)"))
    T = _coordinate_type(xspan)
    xsp = (convert(T, xspan[1]), convert(T, xspan[2]))
    xsp[1] < xsp[2] || throw(ArgumentError("xspan must increase, got $xsp"))
    chain = Chain(Dense(2, S, activation), Dense(S, D, identity; use_bias = false))
    symbolic = SymbolicNeuralNetwork(chain)
    trace = TensorBSplineAnsatz(; degree, ncells, xspan = xsp, x_boundary = Free())
    order = quadrature_order === nothing ? _default_quadrature_order(degree_of(trace)) + 1 :
            Int(quadrature_order)
    order ≥ 1 || throw(ArgumentError("quadrature_order must be at least 1, got $order"))
    q = TensorProductQuadrature(basis(trace);
        nq = (order, quadrature_order_of(trace)[2]))
    n = length(Base.first(flatten(params(NeuralNetwork(chain, T)))))
    last_layer = (n - S * D + 1):n
    free = mode === :all ? n : length(last_layer)
    start = mode === :all ? 1 : Base.first(last_layer)
    field, gradient = network_functions(symbolic, D)
    return NetworkAnsatz{T, typeof(chain), typeof(field), typeof(gradient), typeof(trace),
        typeof(q)}(chain, xsp, mode, trace, q, field, gradient, Int(S), Int(D), n,
        free, start)
end

# The degree of the trace basis in its time direction, which is the one the rule's order follows.
degree_of(a::TensorBSplineAnsatz) = degree(a)[1]

# The number of Gauss nodes per cell the trace basis' rule takes in each direction.
function quadrature_order_of(a::TensorBSplineAnsatz)
    rules = quadratures(quadrature(a))
    return (rules[1].nq, rules[2].nq)
end

"""
    network_functions(symbolic, D)

The two executable functions of a network of `D` outputs, generated once from its symbolic
derivatives.

`field(input, parameters)` returns the `3D` values `[u; ∂_{t̂}u; ∂ₓu]` at one point, and
`gradient(input, parameters)` their derivative with respect to every parameter, a `3D × NP` matrix.
Both take the parameters as the flat vector of `NeuralNetworkParameters`, whose order is `L1.W`,
`L1.b`, `L2.W`, and both read the input as `[t̂, x]`.

The derivatives are those of `SymbolicNeuralNetworks`, whose `Jacobian` and `derivative` are not
exported and are imported by name here.
"""
function network_functions(symbolic::SymbolicNeuralNetwork, D::Integer)
    slopes = derivative(Jacobian(symbolic))
    components = symbolic.model(symbolic.input, params(symbolic))
    equations = [components[d] for d in 1:D]
    times = [slopes[d, 1] for d in 1:D]
    spaces = [slopes[d, 2] for d in 1:D]
    values = vcat(equations, times, spaces)
    return (build_flat_function(values, symbolic),
        build_flat_function(flat_parameter_gradient(values, symbolic), symbolic))
end

# A function of `build_flat_function` lays its flat parameter vector out into a new parameter set and
# allocates its result on every call. A point evaluation calls the in-place kernel it wraps instead,
# `kernel!(out, input, parameters, 1)`, from a parameter set the block keeps and into the block's own
# scratch. The layout and the kernel are fields of `SymbolicNeuralNetworks`' `FlatParameterFunction`
# and `InPlaceBatchedFunction`, which that package does not declare public.
flat_layout(f) = f.layout
flat_kernel!(f) = f.f.kernel!

# The parameter set of element type `ST` that the network's kernels read.
function parameter_set(a::NetworkAnsatz, ::Type{ST}) where {ST}
    return unflatten(flat_layout(a.field), zeros(ST, a.np))
end

"""The space interval of the slab."""
xspan(ansatz::NetworkAnsatz) = ansatz.xspan

"""
trace_basis(ansatz::NetworkAnsatz)

The trace basis of the network's multipliers: a [`TensorBSplineAnsatz`](@ref) whose functions the
coupling uses as its multipliers and on whose rules its tables are built.
"""
function trace_basis(ansatz::NetworkAnsatz)
    return ansatz.trace
end

"""The rule the network evaluates its field on, which is not the trace basis'."""
quadrature(ansatz::NetworkAnsatz) = ansatz.quadrature

"""The number of outputs of the network, which is the number of components of the field."""
noutputs(ansatz::NetworkAnsatz) = ansatz.D

"""The activation of the hidden layer."""
activation_function(ansatz::NetworkAnsatz) = layers(ansatz.chain)[1].σ

"""The number of the network's parameters, in the order `L1.W`, `L1.b`, `L2.W`."""
nparameters(ansatz::NetworkAnsatz) = ansatz.np

"""
    fit_atoms(w, ansatz::NetworkAnsatz)

The atoms a fitted parameter vector selected, as the `(w₁, w₂, b)` triples of its hidden units.
"""
function fit_atoms(w::AbstractVector, ansatz::NetworkAnsatz)
    S = ansatz.S
    return [(w[j], w[S + j], w[2S + j]) for j in 1:S]
end

"""
    output_weights(w, ansatz::NetworkAnsatz, d)

The output weights of component `d` of a fitted parameter vector, one per hidden unit.
"""
function output_weights(w::AbstractVector, ansatz::NetworkAnsatz, d::Integer)
    S, D = ansatz.S, ansatz.D
    return [w[3S + (j - 1) * D + d] for j in 1:S]
end

"""
    nfree(ansatz::NetworkAnsatz)

The number of a network's own unknowns: its free parameters, which neither the coupling nor the
problem changes.
"""
nfree(ansatz::NetworkAnsatz) = ansatz.nfree

"""
    check_ansatz(problem, ansatz)

Reject a network that the problem cannot support, when the integrator is built.

The element type must be the problem's, the network must have one output per component of the field,
and the `xspan` must be the problem's. Each refusal is an `ArgumentError` that names the two values
it compared, and the element type is compared first, so a pair that mismatches in both hears about
the type.
"""
function check_ansatz(problem::LagrangianPDEProblem{PT}, ansatz::NetworkAnsatz{T}) where {
        PT, T}
    T === PT || throw(ArgumentError(
        "the ansatz element type $T does not match the problem's $PT"))
    ansatz.D == components(problem) || throw(ArgumentError(
        "the network has $(ansatz.D) output(s) where the problem has " *
        "$(components(problem)) component(s)"))
    xspan(ansatz) == xspan(problem) || throw(ArgumentError(
        "the ansatz xspan $(xspan(ansatz)) does not match the problem's $(xspan(problem))"))
    return nothing
end

is_linear_in_coefficients(::NetworkAnsatz) = false

has_affine_step(a::NetworkAnsatz) = a.mode === :last_layer

default_fit(::NetworkAnsatz) = OrthogonalGreedy()

"""
    fit!(w, method::OrthogonalGreedy, ansatz::NetworkAnsatz, sample, cache)

Fit the network to a slab's sampled target, by the greedy fit of
[`fit_network!`](@ref) on the workspace of the cache's block, and write the fitted atoms into `w`
in the order `L1.W`, `L1.b`, `L2.W`: the two hidden weights and the bias of every selected atom, and
the output weight of every component behind it. A unit the selection passed over leaves its
parameters at zero.

The output weights are the ones the fit returns: the projection is orthogonal to the atoms already
taken, so its coefficients are those of the atoms and not of the network, and the fit's triangular
solve turns them into the output weights of a hidden layer that is already fixed.
"""
function fit!(w::AbstractVector{ST}, method::OrthogonalGreedy, ansatz::NetworkAnsatz,
        sample::AbstractArray{ST, 3}, c::SpaceTimeGalerkinCache) where {ST}
    f = c.ansatz.fit
    n = fit_network!(f, method, sample, c.ansatz.tabulation, ansatz.S, ansatz.D,
        activation_function(ansatz))
    S, D = ansatz.S, ansatz.D
    fill!(w, zero(ST))
    for j in 1:n
        k = f.selected[j]
        w[j], w[S + j], w[2S + j] = f.atoms[k, 1], f.atoms[k, 2], f.atoms[k, 3]
        for d in 1:D
            w[3S + (j - 1) * D + d] = f.coefficients[j, d]
        end
    end
    return w
end

"""
    fit!(w, method::TrainedHiddenLayer, ansatz::NetworkAnsatz, sample, cache)

Fit the hidden layer and the output layer of a `:last_layer` network to a slab's sampled target, by
[`fit_trained!`](@ref) on the workspace of the cache's block. `sample` is the named tuple
`(; interior, faces)` that [`fit_sample`](@ref) returns for the method: the target at the nodes of
the ansatz's own rule, of size `(Qt, Qx, D)`, and at the four surfaces of the slab, one row per
point.

The mode is `:last_layer` alone: in `:all` mode Newton determines every parameter, so a fit that
trained the hidden layer as well would hand the same weights a second owner.
"""
function fit!(w::AbstractVector{ST}, method::TrainedHiddenLayer, ansatz::NetworkAnsatz,
        sample::NamedTuple, c::SpaceTimeGalerkinCache) where {ST}
    ansatz.mode === :last_layer || throw(ArgumentError(
        "TrainedHiddenLayer needs an ansatz in :last_layer mode; Newton solves every layer in " *
        ":$(ansatz.mode) mode"))
    fit_trained!(c.ansatz.train, method, w, ansatz, sample, c.ansatz)
    return w
end

"""
    fit_sample(method, interior, target, cache, ansatz, problem, sol)

The sample a fit method takes, from the target's `interior` sample at the nodes of the ansatz's own
rule. A fit that takes the interior sample alone takes `interior`. A
[`TrainedHiddenLayer`](@ref) takes the named tuple `(; interior, faces)`, whose `faces` is the
target at the four surfaces of the slab, one row per point, at the point sets of
[`trace_basis`](@ref)`(ansatz)`: the two surfaces in time at the space rule, then the two in space
at the time rule.
"""
fit_sample(::SlabFitMethod, interior, target, c, ansatz, problem, sol) = interior

function fit_sample(::TrainedHiddenLayer, interior::AbstractArray{ST, 3}, target, c,
        ansatz::NetworkAnsatz, problem::LagrangianPDEProblem, sol) where {ST}
    xt, tt = c.ansatz.trace_tabulation.space, c.ansatz.trace_tabulation.time
    qxt, qtt = nquadrature(xt), nquadrature(tt)
    faces = zeros(ST, 2 * qxt + 2 * qtt, ansatz.D)
    row = 0
    for t̂ in (zero(ST), one(ST)), r in 1:qxt

        row += 1
        target_point!(view(faces, row, :), target, c, ansatz, problem, sol, t̂, xt.node[r])
    end
    for r in 1:2, q in 1:qtt

        row += 1
        target_point!(view(faces, row, :), target, c, ansatz, problem, sol, tt.node[q],
            xspan(ansatz)[r])
    end
    return (; interior, faces)
end

"""
    target_point!(values, target, cache, ansatz, problem, sol, t̂, x)

The `D` components of `target` at one point of the slab, whose physical time is `t₀ + h t̂`.

The four targets of the guess each answer at a point, as they answer on the slab's quadrature:
[`InitialCondition`](@ref) is the problem's initial condition, [`PreviousSlab`](@ref) the previous
slab's field continued beyond its top, and [`ExactSolution`](@ref) and [`TargetFunction`](@ref) are
read at the physical time of the point.
"""
function target_point!(values::AbstractVector{ST}, ::InitialCondition,
        c::SpaceTimeGalerkinCache, ansatz::NetworkAnsatz, problem::LagrangianPDEProblem,
        sol, t̂, x) where {ST}
    initial = problem.ics_function([x], parameters(problem))
    for d in eachindex(values)
        values[d] = convert(ST, initial.u[1, d])
    end
    return nothing
end

function target_point!(values::AbstractVector{ST}, ::ExactSolution,
        c::SpaceTimeGalerkinCache, ansatz::NetworkAnsatz, problem::LagrangianPDEProblem,
        sol, t̂, x) where {ST}
    return point_target!(values, problem.solutions.u, problem, sol, t̂, x)
end

function target_point!(values::AbstractVector{ST}, target::TargetFunction,
        c::SpaceTimeGalerkinCache, ansatz::NetworkAnsatz, problem::LagrangianPDEProblem,
        sol, t̂, x) where {ST}
    return point_target!(values, target.f, problem, sol, t̂, x)
end

function target_point!(values::AbstractVector{ST}, ::PreviousSlab,
        c::SpaceTimeGalerkinCache, ansatz::NetworkAnsatz, problem::LagrangianPDEProblem,
        sol, t̂, x) where {ST}
    if c.steps == 0
        return target_point!(values, InitialCondition(), c, ansatz, problem, sol, t̂, x)
    end
    s = c.ansatz
    evaluate_at!(s, one(ST), convert(ST, x))
    for d in eachindex(values)
        values[d] = s.values[d] + convert(ST, t̂) * s.values[ansatz.D + d]
    end
    return nothing
end

# One point of a pointwise target `f(t, x, params)`, converted to the element type.
function point_target!(values::AbstractVector{ST}, f,
        problem::LagrangianPDEProblem, sol, t̂, x) where {ST}
    value = f(slab_start(problem, sol) + timestep(problem) * t̂, x, parameters(problem))
    for d in eachindex(values)
        values[d] = convert(ST, value isa Number ? value : value[d])
    end
    return nothing
end

"""
    NetworkPoints{T}

A set of space points at which a network block evaluates its field: the two surfaces of the slab at
the nodes of the space rule, and the output grid.
"""
struct NetworkPoints{T}
    points::Vector{T}
end

"""The points of an evaluation, at which the core applies the coupling's transform of the field."""
evaluation_points(e::NetworkPoints) = e.points

"""
    NetworkSlab{ST, T, A}

The block of a [`NetworkAnsatz`](@ref) in the slab cache: what the ansatz keeps of one slab.

`params` is the network's parameter vector of the slab, in the order `L1.W`, `L1.b`, `L2.W`, and
`residual` its interior weak form, one row per free parameter. `u`, `v` and `w` hold the field and
its derivatives at the nodes of the ansatz's own rule, and `Ju`, `Jv` and `Jw` their derivative with
respect to each free parameter there. `bottom`, `top`, `left` and `right` hold the field and its
parameter derivative at the four surfaces of the slab, on the trace basis' rules, which is where the
coupling's rows read them. `tabulation` is the ansatz's own rule and `trace_tabulation` the trace
basis', which the coupling's tables are built on; `output` and `trace` are the two point sets the
core evaluates the top of the slab at, and `fit` the workspace of the greedy fit.

`values` and `gradient` are the scratch of one evaluation — the `3D` values of the field, its two
derivatives and their parameter derivative — `input` its two-entry argument, and `parameter_set`
the parameters laid out as the network's kernels read them, which every evaluation refreshes from
`params`. `train` is the workspace of the [`TrainedHiddenLayer`](@ref) fit, which trains the hidden
layer in place.
"""
struct NetworkSlab{ST, T, A <: NetworkAnsatz{T}, P}
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
    top_u::Matrix{ST}
    left_u::Matrix{ST}
    right_u::Matrix{ST}
    bottom_J::Array{ST, 3}
    top_J::Array{ST, 3}
    left_J::Array{ST, 3}
    right_J::Array{ST, 3}
    lambda_bottom::Matrix{ST}
    lambda_top::Matrix{ST}
    mu_left::Matrix{ST}
    mu_right::Matrix{ST}
    tabulation::SlabTabulation{T}
    trace_tabulation::SlabTabulation{T}
    output::NetworkPoints{T}
    trace::NetworkPoints{T}
    input::Vector{ST}
    values::Vector{ST}
    gradient::Matrix{ST}
    parameter_set::P
    fit::NetworkFit{ST}
    train::TrainedFit{ST}
end

"""
    ansatz_data(ansatz, problem, ST)

The ansatz's block of the slab cache, built once per integrator.
"""
function ansatz_data(a::NetworkAnsatz{T}, problem::LagrangianPDEProblem,
        ::Type{ST}) where {T, ST}
    tabulation = SlabTabulation(a.quadrature)
    trace_tabulation = SlabTabulation(quadrature(a.trace))
    qt, qx = nquadrature(tabulation.time), nquadrature(tabulation.space)
    qtt, qxt = nquadrature(trace_tabulation.time), nquadrature(trace_tabulation.space)
    nf, D = a.nfree, a.D
    zero3(q1, q2) = zeros(ST, q1, q2, D)
    zeroJ(q1, q2) = zeros(ST, q1, q2, D, nf)
    ps = parameter_set(a, ST)
    return NetworkSlab{ST, T, typeof(a), typeof(ps)}(a,
        zeros(ST, a.np), zeros(ST, nf),
        zero3(qt, qx), zero3(qt, qx), zero3(qt, qx),
        zeroJ(qt, qx), zeroJ(qt, qx), zeroJ(qt, qx),
        zeros(ST, qxt, D), zeros(ST, qxt, D), zeros(ST, qtt, D), zeros(ST, qtt, D),
        zeros(ST, qxt, D, nf), zeros(ST, qxt, D, nf), zeros(ST, qtt, D, nf),
        zeros(ST, qtt, D, nf),
        zeros(ST, qxt, D), zeros(ST, qxt, D), zeros(ST, qtt, D), zeros(ST, qtt, D),
        tabulation, trace_tabulation,
        NetworkPoints(collect(T, xgrid(problem))),
        NetworkPoints(collect(T, trace_tabulation.space.node)),
        zeros(ST, 2), zeros(ST, 3D), zeros(ST, 3D, a.np), ps, NetworkFit(ST),
        TrainedFit(ST))
end

"""
    ansatz_data_type(ansatz, problem, ST)

The type of [`ansatz_data`](@ref)`(ansatz, problem, ST)`, computed without building the block.
"""
function ansatz_data_type(a::NetworkAnsatz{T}, ::LagrangianPDEProblem,
        ::Type{ST}) where {T, ST}
    return NetworkSlab{ST, T, typeof(a), typeof(parameter_set(a, ST))}
end

"""The normalised time nodes and the space nodes of the slab's own quadrature."""
function quadrature_points(s::NetworkSlab)
    return (s.tabulation.time.node, s.tabulation.space.node)
end

"""The evaluation of the ansatz at the output grid."""
output_evaluation(s::NetworkSlab) = s.output

"""The evaluation of the ansatz at the nodes of the trace basis' space rule."""
trace_evaluation(s::NetworkSlab) = s.trace

"""
    evaluate_at!(slab, t̂, x)

Evaluate the field, its two derivatives and their parameter derivative at one point into the block's
scratch, without allocating.
"""
@inline function evaluate_at!(s::NetworkSlab, t̂, x)
    a = s.ansatz
    s.input[1] = t̂
    s.input[2] = x
    unflatten!(s.parameter_set, flat_layout(a.field), s.params)
    flat_kernel!(a.field)(s.values, s.input, s.parameter_set, 1)
    flat_kernel!(a.gradient)(s.gradient, s.input, s.parameter_set, 1)
    return nothing
end

"""
    slab_field!(u, v, w, slab, x, bottom, nf, h)

The field of the slab that the solver's vector `x` describes, at the nodes of the ansatz's own
quadrature and at the four surfaces of the slab: the free parameters from `x`, the others as the
block holds them, and the field, its derivatives and their parameter derivative from the network.

`v` is the derivative in physical time, `∂_{t̂}u / h`, and the parameter derivative of `v` carries
the same `1/h`. A network fixes no bottom row, so `bottom` is `nothing`.
"""
function slab_field!(
        u::AbstractArray{ST, 3}, v::AbstractArray{ST, 3}, w::AbstractArray{ST, 3},
        s::NetworkSlab{ST}, x::AbstractVector{ST}, bottom, nf::Integer,
        h) where {ST}
    a = s.ansatz
    for k in 1:nf
        s.params[a.first_free + k - 1] = x[k]
    end
    hh = convert(ST, h)
    D = a.D
    tq, xq = s.tabulation.time, s.tabulation.space
    for q in 1:nquadrature(tq), r in 1:nquadrature(xq)

        evaluate_at!(s, tq.node[q], xq.node[r])
        for d in 1:D
            u[q, r, d] = s.values[d]
            v[q, r, d] = s.values[D + d] / hh
            w[q, r, d] = s.values[2D + d]
            for k in 1:nf
                p = a.first_free + k - 1
                s.Ju[q, r, d, k] = s.gradient[d, p]
                s.Jv[q, r, d, k] = s.gradient[D + d, p] / hh
                s.Jw[q, r, d, k] = s.gradient[2D + d, p]
            end
        end
    end
    tt, xt = s.trace_tabulation.time, s.trace_tabulation.space
    left, right = xspan(a)
    edge!(s.bottom_u, s.bottom_J, s, Iterators.repeated(zero(ST)), xt.node)
    edge!(s.top_u, s.top_J, s, Iterators.repeated(one(ST)), xt.node)
    edge!(s.left_u, s.left_J, s, tt.node, Iterators.repeated(convert(ST, left)))
    edge!(s.right_u, s.right_J, s, tt.node, Iterators.repeated(convert(ST, right)))
    return nothing
end

# The field and its parameter derivative at one set of points of a surface, one row per point.
function edge!(u::AbstractMatrix{ST}, J::AbstractArray{ST, 3}, s::NetworkSlab{ST},
        times, spaces) where {ST}
    a = s.ansatz
    D, nf = a.D, a.nfree
    for (r, (t̂, x)) in enumerate(zip(times, spaces))
        evaluate_at!(s, convert(ST, t̂), convert(ST, x))
        for d in 1:D
            u[r, d] = s.values[d]
            for k in 1:nf
                J[r, d, k] = s.gradient[d, a.first_free + k - 1]
            end
        end
    end
    return nothing
end

"""
    slab_weak_form!(slab, ∂u, ∂v, ∂w, h)

The interior weak form of the slab, one row per free parameter: the Euler–Lagrange residual at the
nodes of the ansatz's own quadrature, weighted by the derivative of the field with respect to that
parameter, which is the test function a network's rows carry.
"""
function slab_weak_form!(s::NetworkSlab{ST}, ∂u, ∂v, ∂w, h) where {ST}
    residual = s.residual
    fill!(residual, zero(ST))
    hh = convert(ST, h)
    D, nf = s.ansatz.D, length(residual)
    tq, xq = s.tabulation.time, s.tabulation.space
    for q in 1:nquadrature(tq), r in 1:nquadrature(xq)

        ω = hh * tq.weight[q] * xq.weight[r]
        for d in 1:D
            ζu = ω * ∂u[q, r, d]
            ζv = ω * ∂v[q, r, d]
            ζw = ω * ∂w[q, r, d]
            for k in 1:nf
                residual[k] += ζu * s.Ju[q, r, d, k] + ζv * s.Jv[q, r, d, k] +
                               ζw * s.Jw[q, r, d, k]
            end
        end
    end
    return nothing
end

"""
    free_residual!(b, slab, nf)

The interior weak form, into the first `nf` entries of the solver's vector, which are the network's
free parameters.
"""
function free_residual!(b::AbstractVector, s::NetworkSlab, nf::Integer)
    for k in 1:nf
        b[k] = s.residual[k]
    end
    return nothing
end

"""A network carries no top row of coefficients: the guess continues its parameter vector."""
function keep_top!(::NetworkSlab)
    return nothing
end

"""
    top_values!(u, v, w, slab, evaluation, h)

The field and its two derivatives at the top of the slab, `t̂ = 1`, at the points of `evaluation`,
into `u[i, d]`, `v[i, d]` and `w[i, d]`. `∂ₜu` is the derivative in the normalised time divided by
the slab's `h`.
"""
function top_values!(u::AbstractMatrix{ST}, v::AbstractMatrix{ST}, w::AbstractMatrix{ST},
        s::NetworkSlab{ST}, evaluation::NetworkPoints, h) where {ST}
    hh = convert(ST, h)
    D = s.ansatz.D
    fill!(u, zero(ST))
    fill!(v, zero(ST))
    fill!(w, zero(ST))
    for i in eachindex(evaluation.points)
        evaluate_at!(s, one(ST), convert(ST, evaluation.points[i]))
        for d in 1:D
            u[i, d] = s.values[d]
            v[i, d] = s.values[D + d] / hh
            w[i, d] = s.values[2D + d]
        end
    end
    return nothing
end

"""
    set_parameters!(slab, w)

Write a whole parameter vector of the network into the block, in the order `L1.W`, `L1.b`, `L2.W`.
"""
function set_parameters!(s::NetworkSlab, w::AbstractVector)
    copyto!(s.params, w)
    return nothing
end

"""
    copy_ansatz_data!(cache_data, data)

Copy the parameters the solver does not free from the cache of the problem's element type into the
cache the solver works in.

A `:last_layer` step frees the output weights alone and leaves the hidden layer where the fit put
it, so the field of such a step and its Jacobian read the fixed parameters. A block of another
element type holds its parameters at zero, and a Jacobian evaluated there would be zero with them.
The free parameters come from the solver's vector on every evaluation, so they are not copied.
"""
function copy_ansatz_data!(c::NetworkSlab{ST}, d::NetworkSlab) where {ST}
    a = d.ansatz
    lo = a.first_free
    hi = lo + a.nfree - 1
    for i in eachindex(c.params)
        (lo ≤ i ≤ hi) && continue
        c.params[i] = convert(ST, d.params[i])
    end
    return nothing
end

# The parameters a `:last_layer` network's step holds fixed are the hidden layer's weights and
# biases, `L1.W` then `L1.b`, which are the `first_free - 1` entries before the free output
# weights: a guess that retrains the hidden layer moves the step's Jacobian.
held_parameters(s::NetworkSlab) = view(s.params, 1:(s.ansatz.first_free - 1))

"""
    nunknowns(ansatz::NetworkAnsatz, coupling, problem)

The number of a network's own unknowns of one slab: its free parameters, which neither the coupling
nor the problem changes.
"""
function nunknowns(ansatz::NetworkAnsatz, coupling::SlabCoupling,
        problem::LagrangianPDEProblem)
    return nfree(ansatz)
end

"""
    initsolver(solvermethod, method, caches)

The network family solves its slab with a rank-revealing linear solve.

A network's rows are its parameters, and the constraint rows of a multiplier coupling depend on the
field alone, so the Jacobian is a saddle point whose rank is at most the number of parameters plus
the number of constraints, and the constraint block reaches only the output weights in `:last_layer`
mode. The minimum-norm step of a complete orthogonal factorization solves a consistent system exactly
and drops the directions the rank does not carry, where a factorisation without rank detection loses
the descent direction and raises on the zero pivot.
"""
function initsolver(solvermethod::Newton,
        method::SpaceTimeGalerkin{<:NetworkAnsatz}, caches::CacheDict; kwargs...)
    options = merge((; linear_solver_method = SimpleSolvers.LapackPivotedQR()),
        values(kwargs))
    return invoke(initsolver,
        Tuple{NonlinearSolverMethod, GeometricMethod, CacheDict},
        solvermethod, method, caches; options...)
end

"""
    evaluate!(sample, target, cache, ansatz::NetworkAnsatz, problem, sol)

Fill `sample`, of size `(Qt, Qx, D)`, with the value of `target` at the nodes of the ansatz's own
quadrature, of which `sol` fixes the physical time.

[`InitialCondition`](@ref) is the problem's initial condition, constant in the normalised time;
[`PreviousSlab`](@ref) is the previous slab's field continued beyond its top by its time derivative
there, and the initial condition before the first slab; and [`ExactSolution`](@ref) and
[`TargetFunction`](@ref) are read at the physical time of each node.
"""
function evaluate!(sample::AbstractArray{ST, 3}, ::InitialCondition,
        c::SpaceTimeGalerkinCache, ansatz::NetworkAnsatz,
        problem::LagrangianPDEProblem, sol) where {ST}
    initial = initial_condition_sample(ST, problem, trace_basis(ansatz))
    for d in axes(sample, 3), r in axes(sample, 2), q in axes(sample, 1)
        sample[q, r, d] = initial[r, d]
    end
    return nothing
end

function evaluate!(sample::AbstractArray{ST, 3}, ::PreviousSlab,
        c::SpaceTimeGalerkinCache, ansatz::NetworkAnsatz,
        problem::LagrangianPDEProblem, sol) where {ST}
    if c.steps == 0
        return evaluate!(sample, InitialCondition(), c, ansatz, problem, sol)
    end
    s = c.ansatz
    D = size(sample, 3)
    for r in axes(sample, 2)
        evaluate_at!(s, one(ST), s.tabulation.space.node[r])
        for d in 1:D, q in axes(sample, 1)

            t̂ = convert(ST, s.tabulation.time.node[q])
            # the displacement beyond the previous top, in the normalised time, times the slope there
            sample[q, r, d] = s.values[d] + t̂ * s.values[D + d]
        end
    end
    return nothing
end

function evaluate!(sample::AbstractArray{ST, 3}, target::TargetFunction,
        c::SpaceTimeGalerkinCache, ansatz::NetworkAnsatz,
        problem::LagrangianPDEProblem, sol) where {ST}
    return network_target_values!(sample, target.f, c.ansatz, problem, sol)
end

function evaluate!(sample::AbstractArray{ST, 3}, ::ExactSolution,
        c::SpaceTimeGalerkinCache, ansatz::NetworkAnsatz,
        problem::LagrangianPDEProblem, sol) where {ST}
    return network_target_values!(sample, problem.solutions.u, c.ansatz, problem, sol)
end

# A pointwise target `f(t, x, params)` at the nodes of the block's own rule, in physical time,
# converted to the problem's element type, one point at a time.
function network_target_values!(sample::AbstractArray{ST, 3}, f, s::NetworkSlab,
        problem::LagrangianPDEProblem, sol) where {ST}
    tq, xq = s.tabulation.time, s.tabulation.space
    for q in 1:nquadrature(tq), r in 1:nquadrature(xq)

        point_target!(view(sample, q, r, :), f, problem, sol, tq.node[q], xq.node[r])
    end
    return nothing
end

"""
    apply_guess!(cache, guess, ansatz::NetworkAnsatz, problem, sol)

Fill the solver's vector of a network slab from the method's initial guess.

A continuation carries the previous slab's parameters into the new one, which the block still holds
where the previous step left them; at the first slab there is nothing to carry, and the guess falls
back to `FitTo(InitialCondition())`. A `FitTo` samples its target at the nodes of the ansatz's own
rule, samples it at the four surfaces of the slab as well where the fit takes them
([`fit_sample`](@ref)), fits the network to it, and takes the free part of the fitted parameters as
its unknowns.
"""
function apply_guess!(c::SpaceTimeGalerkinCache{ST}, guess::Continuation,
        ansatz::NetworkAnsatz, problem::LagrangianPDEProblem, sol) where {ST}
    if c.steps == 0
        apply_guess!(c, FitTo(InitialCondition()), ansatz, problem, sol)
    else
        for k in 1:nfree(ansatz)
            c.x[k] = c.ansatz.params[ansatz.first_free + k - 1]
        end
    end
    return nothing
end

function apply_guess!(c::SpaceTimeGalerkinCache{ST}, guess::FitTo,
        ansatz::NetworkAnsatz, problem::LagrangianPDEProblem, sol) where {ST}
    tnodes, xnodes = quadrature_points(c.ansatz)
    D = ansatz.D
    sample = zeros(ST, length(tnodes), length(xnodes), D)
    evaluate!(sample, guess.target, c, ansatz, problem, sol)
    w = copy(c.ansatz.params)
    fitmethod = fit_method(guess, ansatz)
    fit!(w, fitmethod, ansatz,
        fit_sample(fitmethod, sample, guess.target, c, ansatz, problem, sol), c)
    set_parameters!(c.ansatz, w)
    for k in 1:nfree(ansatz)
        c.x[k] = w[ansatz.first_free + k - 1]
    end
    return nothing
end

function Base.show(io::IO, ansatz::NetworkAnsatz{T}) where {T}
    print(io,
        "NetworkAnsatz{$T} with $(ansatz.S) hidden units and $(ansatz.D) output(s), ",
        "$(ansatz.np) parameters of which $(ansatz.nfree) are free, mode $(ansatz.mode), ",
        "x ∈ $(ansatz.xspan)")
end

# =====================================================================================
# The fixed-boundary network: the parameters of a time level, a bubble on the slab, no
# multipliers. The coupling side of the method is in `couplings/fixed_boundary.jl`.
# =====================================================================================

"""The two coordinate systems of the bubble's hidden layer."""
const BUBBLE_FORMS = (:tsr, :abg)

"""
    gauss_legendre(n, T)

The `n`-point Gauss–Legendre rule on `[0, 1]`: its nodes and its weights, by the symmetric
tridiagonal eigenproblem whose eigenvalues are the nodes and whose first eigenvector entries squared
are the weights.
"""
function gauss_legendre(n::Integer, ::Type{T}) where {T}
    n ≥ 1 || throw(ArgumentError("a Gauss rule needs at least one node, got $n"))
    β = [T(k) / sqrt(T(4 * k^2) - 1) for k in 1:(n - 1)]
    E = eigen(SymTridiagonal(zeros(T, n), β))
    nodes = (E.values .+ 1) ./ 2
    weights = E.vectors[1, :] .^ 2
    return collect(T, nodes), collect(T, weights)
end

"""
    FixedBoundaryTabulation{T}

The nodes and the weights of a fixed-boundary slab's own rule: `nq[1]` Gauss nodes on the
normalised time `[0, 1]`, and `nq[2]` Gauss nodes on each of `ncells` cells of the space interval.
The space nodes are mapped onto the span when the interval is known, which is what
[`FixedBoundaryTabulation`](@ref)`(a, T)` does.
"""
struct FixedBoundaryTabulation{T}
    tnode::Vector{T}
    tweight::Vector{T}
    xnode::Vector{T}
    xweight::Vector{T}
end

"""
    FixedBoundaryNetwork(; Sx = 8, St = 10, bubble_form = :tsr, mode = :all, xspan,
                          nq = (6, 4), ncells = 20, D = 1)

The fixed-boundary network: the configuration of one time level is the parameter vector of a
spatial network, `φ = (w, b, c)`, and the slab between two levels carries a bubble `Ψ` whose hidden
layer is `St` tanh units. There are no multipliers, and a step solves for the next level's
parameters and the bubble's.

The spatial network is

```math
N(x; \\phi, t) = (1 - \\hat s)\\, g_a(t) + \\hat s\\, g_b(t)
                 + (x - a)(b - x) \\sum_i c_i \\tanh(w_i \\hat s + b_i),
\\qquad \\hat s = \\frac{x - a}{b - a},
```

with `g_a` and `g_b` the problem's side data at the two ends of the span. It carries the data of its
own time level, so the slab's field can equal the data at every node of every step. The hidden layer
serves all `D` components, `Sx` weights, `Sx` biases and `Sx × D` output weights, and one unit's
argument is normalised in space, so the same network lives on any span.

`bubble_form` names the coordinates of the bubble's hidden layer: `:abg` is `(α, β, γ)`, and `:tsr`
is `(θ, s, ρ)` with `(α, β, γ) = (e^ρ cos θ, e^ρ sin θ, −e^ρ s)`. `mode` fixes the unknowns of a
step: `:all` frees the level's parameters, the bubble's hidden layer and its output weights, and
`:last_layer` frees the level's output weights `c` and the bubble's output weights alone, with both
hidden layers held as data.

The slab's own rule is `nq[1]` Gauss nodes in the normalised time and `nq[2]` Gauss nodes on each of
`ncells` space cells, and it is not the problem's output grid.

```julia
ansatz = FixedBoundaryNetwork(Sx = 8, St = 10, xspan = (0.0, 1.0), ncells = 20)
```
"""
struct FixedBoundaryNetwork{T, Q <: FixedBoundaryTabulation{T}} <: SpaceTimeAnsatz
    xspan::NTuple{2, T}
    Sx::Int
    St::Int
    D::Int
    bubble_form::Symbol
    mode::Symbol
    nq::NTuple{2, Int}
    ncells::Int
    tabulation::Q
end

function FixedBoundaryNetwork(; Sx = 8, St = 10, bubble_form = :tsr, mode = :all, xspan,
        nq = (6, 4), ncells = 20, D = 1)
    length(xspan) == 2 ||
        throw(ArgumentError("xspan must hold two values, got $xspan"))
    length(nq) == 2 && all(q -> q isa Integer && q ≥ 1, nq) || throw(ArgumentError(
        "nq must hold two positive node counts, got $nq"))
    Sx ≥ 1 || throw(ArgumentError("Sx must be at least 1, got $Sx"))
    St ≥ 1 || throw(ArgumentError("St must be at least 1, got $St"))
    D ≥ 1 || throw(ArgumentError("D must be at least 1, got $D"))
    ncells ≥ 1 || throw(ArgumentError("ncells must be at least 1, got $ncells"))
    bubble_form in BUBBLE_FORMS ||
        throw(ArgumentError("bubble_form must be one of $(BUBBLE_FORMS), got :$(bubble_form)"))
    mode in NETWORK_MODES ||
        throw(ArgumentError("mode must be one of $NETWORK_MODES, got :$(mode)"))
    T = _coordinate_type(xspan)
    xa, xb = convert(T, xspan[1]), convert(T, xspan[2])
    xa < xb || throw(ArgumentError("xspan must increase, got ($xa, $xb)"))
    skeleton = FixedBoundaryNetwork{T, FixedBoundaryTabulation{T}}((xa, xb), Int(Sx),
        Int(St), Int(D), Symbol(bubble_form), Symbol(mode), (Int(nq[1]), Int(nq[2])),
        Int(ncells), FixedBoundaryTabulation{T}(T[], T[], T[], T[]))
    return FixedBoundaryNetwork{T, FixedBoundaryTabulation{T}}((xa, xb), Int(Sx), Int(St),
        Int(D), Symbol(bubble_form), Symbol(mode), (Int(nq[1]), Int(nq[2])), Int(ncells),
        FixedBoundaryTabulation(skeleton))
end

"""The slab's own rule, mapped onto the span of the ansatz."""
function FixedBoundaryTabulation(a::FixedBoundaryNetwork{T}) where {T}
    nt, nx = a.nq
    tnode, tweight = gauss_legendre(nt, T)
    xnode, xweight = gauss_legendre(nx, T)
    xa, xb = a.xspan
    span, cells = xb - xa, T(a.ncells)
    nodes = Vector{T}(undef, length(xnode) * a.ncells)
    weights = similar(nodes)
    for j in 1:a.ncells, i in eachindex(xnode)

        k = (j - 1) * length(xnode) + i
        nodes[k] = xa + span * (T(j - 1) + xnode[i]) / cells
        weights[k] = span * xweight[i] / cells
    end
    return FixedBoundaryTabulation{T}(tnode, tweight, nodes, weights)
end

"""The number of a level's parameters: `Sx` weights, `Sx` biases and `Sx × D` output weights."""
nspatial(a::FixedBoundaryNetwork) = 2 * a.Sx + a.Sx * a.D

"""The number of the bubble's parameters: `3St` hidden entries and `St × D` output weights."""
nbubble(a::FixedBoundaryNetwork) = 3 * a.St + a.St * a.D

"""The positions, within a level's parameters, that the mode frees."""
function spatial_free(a::FixedBoundaryNetwork)
    a.mode === :all ? (1:nspatial(a)) : ((2 * a.Sx + 1):(3 * a.Sx))
end

"""The positions, within the bubble's parameters, that the mode frees."""
function bubble_free(a::FixedBoundaryNetwork)
    a.mode === :all ? (1:nbubble(a)) : ((3 * a.St + 1):nbubble(a))
end

"""
    nfree(ansatz::FixedBoundaryNetwork)

The number of the ansatz's own unknowns of one slab: the level's free parameters and the bubble's.
"""
nfree(a::FixedBoundaryNetwork) = length(spatial_free(a)) + length(bubble_free(a))

"""
    free_indices(ansatz::FixedBoundaryNetwork)

The positions of the solver's vector within the whole parameter vector of a slab: the level's
parameters come first, in the order `L1.W`, `L1.b`, `L2.W`, and the bubble's parameters after them,
the hidden block first and the output weights last. Under `:last_layer` the solver's vector holds two
slices of that layout, the level's output weights and the bubble's, so the block maps it through this
index vector rather than through one starting point.
"""
function free_indices(a::FixedBoundaryNetwork)
    return vcat(collect(spatial_free(a)), nspatial(a) .+ collect(bubble_free(a)))
end

"""The whole parameter vector of a slab: the level's parameters, then the bubble's."""
nparameters(a::FixedBoundaryNetwork) = nspatial(a) + nbubble(a)

"""The number of outputs of the network, which is the number of components of the field."""
noutputs(a::FixedBoundaryNetwork) = a.D

"""The space interval of the slab."""
xspan(a::FixedBoundaryNetwork) = a.xspan

"""The rule of the slab's own quadrature."""
quadrature(a::FixedBoundaryNetwork) = a.tabulation

"""
    bubble_chart(α, β, γ, unit = 0)

The `(θ, s, ρ)` coordinates of a hidden unit whose coordinates are `(α, β, γ)`:
`ρ = log √(α² + β²)`, `θ = atan(β, α)` and `s = −γ / e^ρ`.

A unit at `α = β = 0`, that is `r = 0`, has no `(θ, s, ρ)` coordinates — the radius is the
exponential of `ρ`, which is never zero — so converting one raises `ArgumentError` naming the unit
instead of returning a `NaN`.
"""
function bubble_chart(α, β, γ, unit::Integer = 0)
    r = sqrt(α^2 + β^2)
    iszero(r) && throw(ArgumentError(
        "the bubble's unit $unit has α = β = 0, that is r = 0, and no (θ, s, ρ) coordinates: " *
        "exp(ρ) is never zero"))
    return (atan(β, α), -γ / r, log(r))
end

"""
    bubble_values(Θ, St, D, d, t̂, ŝ, form)

The bubble's hidden sum for component `d` at one point, and its derivatives in `t̂` and in `ŝ`:
`Σ_k a_k tanh(α_k t̂ + β_k ŝ + γ_k)`, with `(α, β, γ)` read in the coordinate system of `form`.
"""
@inline function bubble_values(Θ::AbstractVector{T}, St::Integer, D::Integer, d::Integer,
        t̂, ŝ, form::Symbol) where {T}
    m = zero(T)
    mt = zero(T)
    mŝ = zero(T)
    for k in 1:St
        if form === :tsr
            θ = Θ[k]
            s = Θ[St + k]
            ρ = Θ[2 * St + k]
            r = exp(ρ)
            α = r * cos(θ)
            β = r * sin(θ)
            γ = -r * s
        else
            α = Θ[k]
            β = Θ[St + k]
            γ = Θ[2 * St + k]
        end
        a = Θ[3 * St + (k - 1) * D + d]
        t = tanh(α * t̂ + β * ŝ + γ)
        slope = a * (one(t) - t * t)
        m += a * t
        mt += slope * α
        mŝ += slope * β
    end
    return m, mt, mŝ
end

"""
    spatial_values(φ, Sx, D, d, ŝ)

The spatial network's hidden sum for component `d` at `ŝ`, and its derivative in `ŝ`:
`Σ_i c_i tanh(w_i ŝ + b_i)`.
"""
@inline function spatial_values(φ::AbstractVector{T}, Sx::Integer, D::Integer, d::Integer,
        ŝ) where {T}
    m = zero(T)
    mŝ = zero(T)
    for i in 1:Sx
        t = tanh(φ[i] * ŝ + φ[Sx + i])
        c = φ[2 * Sx + (i - 1) * D + d]
        m += c * t
        mŝ += c * φ[i] * (one(t) - t * t)
    end
    return m, mŝ
end

"""
    FixedBoundaryDatum{ST}

What the field needs of the coupling's datum: the bottom level's parameters, and the side data of
the slab at its two end times, at its nodes, and their derivatives in physical time.
"""
struct FixedBoundaryDatum{ST}
    phi::Vector{ST}
    left0::Vector{ST}
    right0::Vector{ST}
    left1::Vector{ST}
    right1::Vector{ST}
    dleft1::Vector{ST}
    dright1::Vector{ST}
end

function FixedBoundaryDatum(::Type{ST}, n::Integer, D::Integer) where {ST}
    side() = zeros(ST, D)
    return FixedBoundaryDatum{ST}(
        zeros(ST, n), side(), side(), side(), side(), side(), side())
end

"""
    FixedBoundaryPoints{T}

A set of space points at which a fixed-boundary block evaluates its field: the output grid, or the
space nodes of the slab's own rule.
"""
struct FixedBoundaryPoints{T}
    points::Vector{T}
end

"""The points of an evaluation, at which the core applies the coupling's transform of the field."""
evaluation_points(e::FixedBoundaryPoints) = e.points

"""
    FixedBoundarySlab{ST, T, A}

The block of a [`FixedBoundaryNetwork`](@ref) in the slab cache: what the ansatz keeps of one slab.

`params` is the whole parameter vector of the slab — the solved level's parameters, then the
bubble's hidden block, then its output weights — and `residual` the action's gradient on the
equations of the step: the derivative with respect to the bottom level's free parameters and to the
bubble's, one row per unknown. `momentum` holds the third block of that gradient, the derivative
with respect to the solved level's free parameters, which is the momentum the step hands on.

`u`, `v` and `w` hold the field and its derivatives at the nodes of the slab's own rule, `Ju`, `Jv`
and `Jw` their derivative with respect to each equation's parameter, and `Jt_u`, `Jt_v` and `Jt_w`
their derivative with respect to each of the solved level's free parameters. `datum` is what the
block keeps of the coupling's datum, which the evaluation at the top of the slab needs; `top_sample`
is the target at `t̂ = 1` on the space nodes, which the fit of the level reads. `fit` is the
workspace of the bubble's fit.
"""
struct FixedBoundarySlab{ST, T, A <: FixedBoundaryNetwork{T}}
    ansatz::A
    params::Vector{ST}
    residual::Vector{ST}
    momentum::Vector{ST}
    u::Array{ST, 3}
    v::Array{ST, 3}
    w::Array{ST, 3}
    Ju::Array{ST, 4}
    Jv::Array{ST, 4}
    Jw::Array{ST, 4}
    Jt_u::Array{ST, 4}
    Jt_v::Array{ST, 4}
    Jt_w::Array{ST, 4}
    datum::FixedBoundaryDatum{ST}
    top_sample::Matrix{ST}
    tabulation::FixedBoundaryTabulation{T}
    output::FixedBoundaryPoints{T}
    trace::FixedBoundaryPoints{T}
    fit::FixedBoundaryFit{ST}
end

"""
    ansatz_data(ansatz, problem, ST)

The ansatz's block of the slab cache, built once per integrator.
"""
function ansatz_data(a::FixedBoundaryNetwork{T}, problem::LagrangianPDEProblem,
        ::Type{ST}) where {T, ST}
    tb = a.tabulation
    qt, qx, D = length(tb.tnode), length(tb.xnode), a.D
    nf, nfp = nfree(a), length(spatial_free(a))
    field3() = zeros(ST, qt, qx, D)
    derivative(n) = zeros(ST, qt, qx, D, n)
    return FixedBoundarySlab{ST, T, typeof(a)}(a, zeros(ST, nparameters(a)),
        zeros(ST, nf), zeros(ST, nfp), field3(), field3(), field3(),
        derivative(nf), derivative(nf), derivative(nf),
        derivative(nfp), derivative(nfp), derivative(nfp),
        FixedBoundaryDatum(ST, nspatial(a), D), zeros(ST, qx, D), tb,
        FixedBoundaryPoints(collect(T, xgrid(problem))),
        FixedBoundaryPoints(collect(T, tb.xnode)), FixedBoundaryFit(ST))
end

"""
    ansatz_data_type(ansatz, problem, ST)

The type of [`ansatz_data`](@ref)`(ansatz, problem, ST)`, computed without building the block.
"""
function ansatz_data_type(a::FixedBoundaryNetwork{T}, ::LagrangianPDEProblem,
        ::Type{ST}) where {T, ST}
    return FixedBoundarySlab{ST, T, typeof(a)}
end

"""The normalised time nodes and the space nodes of the slab's own quadrature."""
function quadrature_points(s::FixedBoundarySlab)
    return (s.tabulation.tnode, s.tabulation.xnode)
end

"""The evaluation of the ansatz at the output grid."""
output_evaluation(s::FixedBoundarySlab) = s.output

"""The evaluation of the ansatz at the space nodes of the slab's own rule."""
trace_evaluation(s::FixedBoundarySlab) = s.trace

"""The bubble's parameters of a block: the tail of its parameter vector."""
function bubble_parameters(s::FixedBoundarySlab)
    a = s.ansatz
    return view(s.params, (nspatial(a) + 1):length(s.params))
end

# A whole parameter vector from a slice of free parameters: the fixed part of `base`, with the
# entries of `inds` replaced by the `length(inds)` values that follow position `off` of `ζ`.
@inline function assemble_parameters(base::AbstractVector, ζ::AbstractVector{T}, inds,
        off::Integer) where {T}
    p = collect(T, base)
    for (k, i) in enumerate(inds)
        p[i] = ζ[off + k]
    end
    return p
end

# The side data of one component at one node of the slab: the two ends' values at the two times of
# the slab and at the node's own time, and the node's time derivative.
@inline function node_sides(data::FixedBoundaryData, d::Integer, q::Integer)
    return (data.ga[1, d], data.gb[1, d], data.ga[end, d], data.gb[end, d],
        data.ga[q + 1, d], data.gb[q + 1, d], data.dga[q + 1, d], data.dgb[q + 1, d])
end

# The side data of one component at the top of the slab.
@inline function top_sides(datum::FixedBoundaryDatum, d::Integer)
    return (datum.left0[d], datum.right0[d], datum.left1[d], datum.right1[d],
        datum.left1[d], datum.right1[d], datum.dleft1[d], datum.dright1[d])
end

"""
    slab_component(ansatz, d, φ0, φ1, Θ, t̂, x, h, sides)

One component of the slab's field at one point, and its two derivatives: `u`, `∂_{t̂}u` and
`∂ₓu`, from the two levels' parameters, the bubble's, and the side data `sides` of that point.

The field is the two levels' networks weighted by `1 − t̂` and `t̂`, the lift `G` that carries the
side data of the slab's own time, and the bubble whose two weights vanish at both ends of the
interface, so that the field equals the side data at `x = a` and `x = b` for every parameter vector.
"""
@inline function slab_component(a::FixedBoundaryNetwork, d::Integer,
        φ0::AbstractVector{T}, φ1::AbstractVector{T}, Θ::AbstractVector{T}, t̂, x, h,
        sides) where {T}
    xa, xb = a.xspan
    L = xb - xa
    ŝ = (x - xa) / L
    bump = (x - xa) * (xb - x)
    slope = xa + xb - 2 * x
    ga0, gb0, ga1, gb1, ga, gb, dga, dgb = sides
    m0, m0ŝ = spatial_values(φ0, a.Sx, a.D, d, ŝ)
    m1, m1ŝ = spatial_values(φ1, a.Sx, a.D, d, ŝ)
    n0 = (1 - ŝ) * ga0 + ŝ * gb0 + bump * m0
    n1 = (1 - ŝ) * ga1 + ŝ * gb1 + bump * m1
    n0x = (gb0 - ga0) / L + slope * m0 + bump * m0ŝ / L
    n1x = (gb1 - ga1) / L + slope * m1 + bump * m1ŝ / L
    ψ, ψt, ψŝ = bubble_values(Θ, a.St, a.D, d, t̂, ŝ, a.bubble_form)
    ra = ga - (1 - t̂) * ga0 - t̂ * ga1
    rb = gb - (1 - t̂) * gb0 - t̂ * gb1
    rat = h * dga + ga0 - ga1
    rbt = h * dgb + gb0 - gb1
    b = t̂ * (1 - t̂)
    u = (1 - t̂) * n0 + t̂ * n1 + (1 - ŝ) * ra + ŝ * rb + b * bump * ψ
    ut = n1 - n0 + (1 - ŝ) * rat + ŝ * rbt + bump * ((1 - 2 * t̂) * ψ + b * ψt)
    ux = (1 - t̂) * n0x + t̂ * n1x + (rb - ra) / L + b * (slope * ψ + bump * ψŝ / L)
    return u, ut, ux
end

"""
    side_derivative(f, t)

The derivative of a scalar side function of time at `t`, by `ForwardDiff` on the function itself.

The lift of a slab differentiates the problem's side data in time, and the data function is the only
source of that derivative: a `bcs_function` that is not generic in its time argument therefore raises
where the integrator is built.
"""
side_derivative(f, t) = ForwardDiff.derivative(f, t)

"""
    spatial_derivative(ansatz, φ, i, d, ŝ, bump, slope, span)

One component of the derivative of a level's network `N(x; φ, t)` in `x`, and of its derivative in
`x`, with respect to parameter `i` of `φ`: the weight, the bias or the output weight of one hidden
unit.
"""
@inline function spatial_derivative(
        a::FixedBoundaryNetwork, φ::AbstractVector{T}, i::Integer,
        d::Integer, ŝ, bump, slope, span) where {T}
    Sx, D = a.Sx, a.D
    if i ≤ Sx
        k = i
        t = tanh(φ[k] * ŝ + φ[Sx + k])
        s = one(t) - t * t
        c = φ[2 * Sx + (k - 1) * D + d]
        return (bump * c * s * ŝ,
            slope * c * s * ŝ + (bump / span) * c * s * (one(t) - 2 * φ[k] * t * ŝ))
    elseif i ≤ 2 * Sx
        k = i - Sx
        t = tanh(φ[k] * ŝ + φ[Sx + k])
        s = one(t) - t * t
        c = φ[2 * Sx + (k - 1) * D + d]
        return (bump * c * s, slope * c * s - (bump / span) * 2 * c * φ[k] * t * s)
    else
        k = (i - 2 * Sx - 1) ÷ D + 1
        t = tanh(φ[k] * ŝ + φ[Sx + k])
        return (bump * t, slope * t + (bump / span) * φ[k] * (one(t) - t * t))
    end
end

"""
    bubble_unit(ansatz, Θ, k)

The `(α, β, γ)` coordinates of hidden unit `k` of a bubble, and its radius `r = e^ρ`, in the
coordinate system of the ansatz's form.
"""
@inline function bubble_unit(a::FixedBoundaryNetwork, Θ::AbstractVector{T}, k::Integer) where {T}
    if a.bubble_form === :tsr
        θ, s, ρ = Θ[k], Θ[a.St + k], Θ[2 * a.St + k]
        r = exp(ρ)
        return (r * cos(θ), r * sin(θ), -r * s, r)
    end
    return (Θ[k], Θ[a.St + k], Θ[2 * a.St + k], one(T))
end

"""
    bubble_derivative(ansatz, Θ, i, d, t̂, ŝ)

The bubble's sum at one point, its two derivatives, and their derivative with respect to the free
parameter `i` of `Θ`: a hidden entry, in the coordinates of the ansatz's form, or an output weight.

The hidden entries' derivatives are taken in `(α, β, γ)` and carried to the form's coordinates by
the chart's own Jacobian, so that a `:tsr` residual is the gradient of the action written in
`(θ, s, ρ)` and its derivative is the chart applied to the equations as well as to the unknowns.
"""
@inline function bubble_derivative(
        a::FixedBoundaryNetwork, Θ::AbstractVector{T}, i::Integer,
        d::Integer, t̂, ŝ) where {T}
    St, D = a.St, a.D
    if i ≤ 3 * St
        k = (i - 1) % St + 1
        which = (i - 1) ÷ St + 1
        ak = Θ[3 * St + (k - 1) * D + d]
        α, β, γ, r = bubble_unit(a, Θ, k)
        t = tanh(α * t̂ + β * ŝ + γ)
        s = one(t) - t * t
        dα, dβ, dγ = ak * s * t̂, ak * s * ŝ, ak * s
        dtα = ak * s * (one(t) - 2 * α * t * t̂)
        dtβ = -2 * ak * α * t * s * ŝ
        dtγ = -2 * ak * α * t * s
        dŝα = -2 * ak * β * t * s * t̂
        dŝβ = ak * s * (one(t) - 2 * β * t * ŝ)
        dŝγ = -2 * ak * β * t * s
        if a.bubble_form === :tsr
            if which == 1
                return (-β * dα + α * dβ, -β * dtα + α * dtβ, -β * dŝα + α * dŝβ)
            elseif which == 2
                return (-r * dγ, -r * dtγ, -r * dŝγ)
            else
                return (α * dα + β * dβ + γ * dγ, α * dtα + β * dtβ + γ * dtγ,
                    α * dŝα + β * dŝβ + γ * dŝγ)
            end
        end
        which == 1 && return (dα, dtα, dŝα)
        which == 2 && return (dβ, dtβ, dŝβ)
        return (dγ, dtγ, dŝγ)
    end
    k = (i - 3 * St - 1) ÷ D + 1
    α, β, γ, _ = bubble_unit(a, Θ, k)
    t = tanh(α * t̂ + β * ŝ + γ)
    s = one(t) - t * t
    return (t, α * s, β * s)
end

"""
    slab_field!(u, v, w, slab, x, bottom, nf, h)

The field of the slab that the solver's vector `x` describes, at the nodes of the slab's own rule:
the free parameters from `x`, the levels' and the bubble's fixed parameters from the block and the
coupling's datum, and the field, its derivatives and their parameter derivative by the network of
the two levels and the bubble. `v` is the derivative in physical time, and the parameter derivative
of `v` carries the same `1 / h`.

The parameter derivative of the field is that of an equation's own parameter: the bottom level's for
the first rows, whose equations the datum's momentum is added to, and the bubble's behind them. The
bottom level's parameters are data of the step, so its derivative is taken in the block's arrays
rather than in the solver's vector, and the solved level's derivative goes to the momentum block.
"""
function slab_field!(
        u::AbstractArray{ST, 3}, v::AbstractArray{ST, 3}, w::AbstractArray{ST, 3},
        s::FixedBoundarySlab{ST}, x::AbstractVector{ST}, bottom, nf::Integer,
        h) where {ST}
    a = s.ansatz
    data = bottom
    for (k, i) in enumerate(free_indices(a))
        s.params[i] = x[k]
    end
    D, nfp = a.D, length(spatial_free(a))
    tb = s.tabulation
    xa, xb = a.xspan
    span = xb - xa
    hh = convert(ST, h)
    free_spatial, free_bubble = spatial_free(a), bubble_free(a)
    φ0, φ1, Θ = data.phi, view(s.params, 1:nspatial(a)), bubble_parameters(s)
    for r in eachindex(tb.xnode), q in eachindex(tb.tnode)

        t̂ = convert(ST, tb.tnode[q])
        xr = convert(ST, tb.xnode[r])
        ŝ = (xr - xa) / span
        bump = (xr - xa) * (xb - xr)
        slope = xa + xb - 2 * xr
        weight = t̂ * (1 - t̂)
        for d in 1:D
            sides = node_sides(data, d, q)
            uu, ut, ux = slab_component(a, d, φ0, φ1, Θ, t̂, xr, hh, sides)
            u[q, r, d] = uu
            v[q, r, d] = ut / hh
            w[q, r, d] = ux
            for (k, i) in enumerate(free_spatial)
                gN, gNx = spatial_derivative(a, φ0, i, d, ŝ, bump, slope, span)
                s.Ju[q, r, d, k] = (1 - t̂) * gN
                s.Jv[q, r, d, k] = -gN / hh
                s.Jw[q, r, d, k] = (1 - t̂) * gNx
                gN1, gNx1 = spatial_derivative(a, φ1, i, d, ŝ, bump, slope, span)
                s.Jt_u[q, r, d, k] = t̂ * gN1
                s.Jt_v[q, r, d, k] = gN1 / hh
                s.Jt_w[q, r, d, k] = t̂ * gNx1
            end
            for (k, i) in enumerate(free_bubble)
                gψ, gψt, gψŝ = bubble_derivative(a, Θ, i, d, t̂, ŝ)
                s.Ju[q, r, d, nfp + k] = weight * bump * gψ
                s.Jv[q, r, d, nfp + k] = bump * ((1 - 2 * t̂) * gψ + weight * gψt) / hh
                s.Jw[q, r, d, nfp + k] = weight * (slope * gψ + bump * gψŝ / span)
            end
        end
    end
    copyto!(s.datum.phi, data.phi)
    for d in 1:D
        s.datum.left0[d], s.datum.right0[d] = data.ga[1, d], data.gb[1, d]
        s.datum.left1[d], s.datum.right1[d] = data.ga[end, d], data.gb[end, d]
        s.datum.dleft1[d], s.datum.dright1[d] = data.dga[end, d], data.dgb[end, d]
    end
    return nothing
end

"""
    slab_weak_form!(slab, ∂u, ∂v, ∂w, h)

The action's gradient on the equations of the step, from the derivatives of the density at the nodes
of the slab's own rule: one row per equation, tested against the derivative of the field with
respect to that equation's parameter, and the third block — the derivative with respect to the solved
level's free parameters — into the block's `momentum` buffer.
"""
function slab_weak_form!(s::FixedBoundarySlab{ST}, ∂u, ∂v, ∂w, h) where {ST}
    a = s.ansatz
    D, nf = a.D, length(s.residual)
    nfp = length(s.momentum)
    hh = convert(ST, h)
    fill!(s.residual, zero(ST))
    fill!(s.momentum, zero(ST))
    tw, xw = s.tabulation.tweight, s.tabulation.xweight
    for q in eachindex(tw), r in eachindex(xw)

        ω = hh * tw[q] * xw[r]
        for d in 1:D
            ζu = ω * ∂u[q, r, d]
            ζv = ω * ∂v[q, r, d]
            ζw = ω * ∂w[q, r, d]
            for k in 1:nf
                s.residual[k] += ζu * s.Ju[q, r, d, k] + ζv * s.Jv[q, r, d, k] +
                                 ζw * s.Jw[q, r, d, k]
            end
            for k in 1:nfp
                s.momentum[k] += ζu * s.Jt_u[q, r, d, k] + ζv * s.Jt_v[q, r, d, k] +
                                 ζw * s.Jt_w[q, r, d, k]
            end
        end
    end
    return nothing
end

"""
    free_residual!(b, slab, nf)

The action's gradient of the equations of the step, into the first `nf` entries of the solver's
vector.
"""
function free_residual!(b::AbstractVector, s::FixedBoundarySlab, nf::Integer)
    for k in 1:nf
        b[k] = s.residual[k]
    end
    return nothing
end

"""A fixed-boundary level carries its parameters in the block: the guess continues them."""
keep_top!(::FixedBoundarySlab) = nothing

"""
    top_values!(u, v, w, slab, evaluation, h)

The field and its two derivatives at the top of the slab, `t̂ = 1`, at the points of `evaluation`.
The bubble's weights vanish there, so the field is the solved level's network, and `∂ₜu` is the
whole field's derivative in the normalised time divided by `h`.
"""
function top_values!(u::AbstractMatrix{ST}, v::AbstractMatrix{ST}, w::AbstractMatrix{ST},
        s::FixedBoundarySlab{ST}, evaluation::FixedBoundaryPoints, h) where {ST}
    a = s.ansatz
    D = a.D
    hh = convert(ST, h)
    φ0 = s.datum.phi
    φ1 = view(s.params, 1:nspatial(a))
    Θ = bubble_parameters(s)
    fill!(u, zero(ST))
    fill!(v, zero(ST))
    fill!(w, zero(ST))
    for i in eachindex(evaluation.points)
        x = convert(ST, evaluation.points[i])
        for d in 1:D
            uu, ut, ux = slab_component(a, d, φ0, φ1, Θ, one(ST), x, hh,
                top_sides(s.datum, d))
            u[i, d] = uu
            v[i, d] = ut / hh
            w[i, d] = ux
        end
    end
    return nothing
end

"""
    copy_ansatz_data!(cache_data, data)

Copy the parameters the solver does not free from the cache of the problem's element type into the
cache the solver works in. Under `:last_layer` those are both hidden layers, which the guess and the
fit place and the step never moves.
"""
function copy_ansatz_data!(c::FixedBoundarySlab{ST}, d::FixedBoundarySlab) where {ST}
    for i in eachindex(c.params)
        i in free_indices(c.ansatz) && continue
        c.params[i] = convert(ST, d.params[i])
    end
    return nothing
end

"""
    nunknowns(ansatz::FixedBoundaryNetwork, coupling, problem)

The number of a fixed-boundary slab's unknowns: the parameters the mode frees, and nothing for the
coupling, which has no multiplier.
"""
function nunknowns(ansatz::FixedBoundaryNetwork, coupling::SlabCoupling,
        problem::LagrangianPDEProblem)
    return nfree(ansatz)
end

"""
    is_linear_in_coefficients(ansatz::FixedBoundaryNetwork)

A fixed-boundary network is nonlinear in its parameters in both modes, so a `Projection()` on it is
refused, and the direct linear solve is not offered.
"""
is_linear_in_coefficients(::FixedBoundaryNetwork) = false

"""
    check_ansatz(problem, ansatz::FixedBoundaryNetwork)

Reject a fixed-boundary network that the problem cannot support, when the integrator is built: the
element type must be the problem's, the network must have one output per component of the field and
the `xspan` must be the problem's.
"""
function check_ansatz(problem::LagrangianPDEProblem{PT}, a::FixedBoundaryNetwork{T}) where {
        PT, T}
    T === PT || throw(ArgumentError(
        "the ansatz element type $T does not match the problem's $PT"))
    a.D == components(problem) || throw(ArgumentError(
        "the network has $(a.D) output(s) where the problem has " *
        "$(components(problem)) component(s)"))
    xspan(a) == xspan(problem) || throw(ArgumentError(
        "the ansatz xspan $(xspan(a)) does not match the problem's $(xspan(problem))"))
    return nothing
end

function Base.show(io::IO, a::FixedBoundaryNetwork{T}) where {T}
    print(io,
        "FixedBoundaryNetwork{$T} with $(a.Sx) spatial units and $(a.St) bubble units, ",
        "$(nparameters(a)) parameters of which $(nfree(a)) are free, mode $(a.mode), ",
        "bubble $(a.bubble_form), x ∈ $(a.xspan)")
end

"""
    bubble_start(ansatz, T)

The bubble's hidden layer a run starts from, in the coordinates of the ansatz's form: the symmetric
grid `α ∈ [-2, 2]`, `β ∈ [2, -2]`, `γ ∈ [-1, 1]` of the fixed-boundary spike, converted to the
form's own coordinates, with the output weights at zero.

At an odd `St` the grid puts its middle unit at `α = β = 0`, which no `(θ, s, ρ)` writes, so a
`:tsr` run of that size raises there rather than solving a different bubble.
"""
function bubble_start(a::FixedBoundaryNetwork, ::Type{T}) where {T}
    St = a.St
    α = collect(T, range(-2, 2; length = St))
    β = collect(T, range(2, -2; length = St))
    γ = collect(T, range(-1, 1; length = St))
    Θ = zeros(T, nbubble(a))
    for k in 1:St
        if a.bubble_form === :tsr
            θ, s, ρ = bubble_chart(α[k], β[k], γ[k], k)
            Θ[k], Θ[St + k], Θ[2 * St + k] = θ, s, ρ
        else
            Θ[k], Θ[St + k], Θ[2 * St + k] = α[k], β[k], γ[k]
        end
    end
    return Θ
end

"""
    apply_guess!(cache, guess::Continuation, ansatz::FixedBoundaryNetwork, problem, sol)

Fill the solver's vector of a fixed-boundary slab from the continuation: the previous level's
parameters as the solved level's guess, and the previous slab's bubble in the form's coordinates. At
the first slab there is no previous level, so the solved level starts at the datum's `φ₀` and the
bubble at [`bubble_start`](@ref).
"""
function apply_guess!(c::SpaceTimeGalerkinCache{ST}, guess::Continuation,
        ansatz::FixedBoundaryNetwork, problem::LagrangianPDEProblem, sol) where {ST}
    nfp = length(spatial_free(ansatz))
    for (k, i) in enumerate(spatial_free(ansatz))
        c.x[k] = c.coupling.phi[i]
    end
    if c.steps == 0
        Θ = bubble_start(ansatz, ST)
        copyto!(c.ansatz.params, 1, c.coupling.phi, 1, nspatial(ansatz))
        copyto!(c.ansatz.params, nspatial(ansatz) + 1, Θ, 1, nbubble(ansatz))
    else
        for (k, i) in enumerate(bubble_free(ansatz))
            c.x[nfp + k] = c.ansatz.params[nspatial(ansatz) + i]
        end
    end
    return nothing
end

"""
    apply_guess!(cache, guess::FitTo, ansatz::FixedBoundaryNetwork, problem, sol)

Fill the solver's vector of a fixed-boundary slab from a fit: the target is sampled at the nodes of
the slab's own rule and at `t̂ = 1` on the space nodes, the fit brings the whole parameter vector to
it, and the free entries of that vector are the unknowns of the step.
"""
function apply_guess!(c::SpaceTimeGalerkinCache{ST}, guess::FitTo,
        ansatz::FixedBoundaryNetwork, problem::LagrangianPDEProblem, sol) where {ST}
    sample = zeros(ST, length(ansatz.tabulation.tnode), length(ansatz.tabulation.xnode),
        ansatz.D)
    evaluate!(sample, guess.target, c, ansatz, problem, sol)
    w = collect(c.ansatz.params)
    fit!(w, fit_method(guess, ansatz), ansatz, sample, c)
    copyto!(c.ansatz.params, w)
    for (k, i) in enumerate(free_indices(ansatz))
        c.x[k] = w[i]
    end
    return nothing
end

"""
    evaluate!(sample, ::InitialCondition, cache, ansatz::FixedBoundaryNetwork, problem, sol)

The problem's initial condition at the nodes of the slab's own rule, constant in the normalised
time, and at `t̂ = 1` on the space nodes.
"""
function evaluate!(sample::AbstractArray{ST, 3}, ::InitialCondition,
        c::SpaceTimeGalerkinCache, ansatz::FixedBoundaryNetwork,
        problem::LagrangianPDEProblem, sol) where {ST}
    initial = problem.ics_function(collect(ansatz.tabulation.xnode), parameters(problem))
    for d in axes(sample, 3), r in axes(sample, 2), q in axes(sample, 1)
        sample[q, r, d] = ST(initial.u[r, d])
    end
    for d in axes(c.ansatz.top_sample, 2), r in axes(c.ansatz.top_sample, 1)

        c.ansatz.top_sample[r, d] = ST(initial.u[r, d])
    end
    return nothing
end

function evaluate!(sample::AbstractArray{ST, 3}, ::ExactSolution,
        c::SpaceTimeGalerkinCache, ansatz::FixedBoundaryNetwork,
        problem::LagrangianPDEProblem, sol) where {ST}
    return fixed_boundary_target!(sample, c.ansatz, problem, sol, problem.solutions.u)
end

function evaluate!(sample::AbstractArray{ST, 3}, target::TargetFunction,
        c::SpaceTimeGalerkinCache, ansatz::FixedBoundaryNetwork,
        problem::LagrangianPDEProblem, sol) where {ST}
    return fixed_boundary_target!(sample, c.ansatz, problem, sol, target.f)
end

"""
    evaluate!(sample, ::Predictor, cache, ansatz::FixedBoundaryNetwork, problem, sol)

The cheap predictor of the slab, at the nodes of the ansatz's rule: [`predictor_sample!`](@ref).
"""
function evaluate!(sample::AbstractArray{ST, 3}, ::Predictor,
        c::SpaceTimeGalerkinCache, ansatz::FixedBoundaryNetwork,
        problem::LagrangianPDEProblem, sol) where {ST}
    return predictor_sample!(sample, c.ansatz, c, problem, sol)
end

# A pointwise target `f(t, x, params)` at the nodes of the slab's own rule, and at the top on the
# space nodes, converted to the block's element type.
function fixed_boundary_target!(sample::AbstractArray{ST, 3}, s::FixedBoundarySlab,
        problem::LagrangianPDEProblem, sol, f) where {ST}
    t₀ = slab_start(problem, sol)
    h = timestep(problem)
    p = parameters(problem)
    nodes = s.tabulation
    for q in axes(sample, 1), r in axes(sample, 2)

        value = f(t₀ + h * nodes.tnode[q], nodes.xnode[r], p)
        for d in axes(sample, 3)
            sample[q, r, d] = ST(value isa Number ? value : value[d])
        end
    end
    for r in axes(s.top_sample, 1)
        value = f(t₀ + h, nodes.xnode[r], p)
        for d in axes(s.top_sample, 2)
            s.top_sample[r, d] = ST(value isa Number ? value : value[d])
        end
    end
    return nothing
end

"""
    check_method(problem, method::SpaceTimeGalerkin{<:FixedBoundaryNetwork})

The refusals of a fixed-boundary method beyond the core's: a target the ansatz has no field map for,
and a density whose velocity derivative the predictor cannot read.
"""
function check_method(problem::LagrangianPDEProblem,
        method::SpaceTimeGalerkin{<:FixedBoundaryNetwork})
    invoke(check_method, Tuple{LagrangianPDEProblem, SpaceTimeGalerkin}, problem, method)
    guess = initial_guess(method)
    if guess isa FitTo
        supported_target(guess.target) || throw(ArgumentError(
            "a FixedBoundaryNetwork takes InitialCondition, ExactSolution, TargetFunction and " *
            "Predictor as targets, got $(nameof(typeof(guess.target)))"))
        fit_method(guess, ansatz(method)) isa BubbleGreedy || throw(ArgumentError(
            "a FixedBoundaryNetwork is fitted by BubbleGreedy, got " *
            "$(nameof(typeof(fit_method(guess, ansatz(method)))))"))
        guess.target isa Predictor && check_predictor_density(problem)
    end
    return nothing
end

"""Whether a target of a [`FixedBoundaryNetwork`](@ref) has a field map."""
supported_target(::InitialCondition) = true
supported_target(::ExactSolution) = true
supported_target(::TargetFunction) = true
supported_target(::Predictor) = true
supported_target(::SlabTarget) = false
