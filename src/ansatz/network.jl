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
the hidden weights `(S, 2)`, the hidden biases `(S,)` and the output weights `(S, D)`, in the order
`L1.W`, `L1.b`, `L2.W` of `NeuralNetworkParameters`, and one hidden layer serves all `D` components.

`mode` fixes which parameters the solver determines: `:all` frees every one of them, and
`:last_layer` frees the output weights alone and leaves the hidden layer where the initial guess put
it, so that the step is affine in its unknowns and one Newton step reaches its tolerance. The
parameters live in the ansatz's block in the slab cache and never in the ansatz, so two integrators
that share one ansatz object give identical results.

The ansatz carries a [`TensorBSplineAnsatz`](@ref) as its trace basis, `trace_basis(ansatz)`, built
from `degree` and `ncells` on the same `xspan` with `Dirichlet()` at both ends. The functions of that
basis are the multipliers of a coupling that imposes the slab's data weakly, and its rule is the one
the coupling's tables and its surface rows are built on. The field itself is evaluated on a rule of
the ansatz's own: `quadrature_order` Gauss nodes per cell in the time direction, and the space grid of
the trace basis, which the cache's trace is kept on. `nothing` takes one node per cell more than the
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
    trace = TensorBSplineAnsatz(; degree, ncells, xspan = xsp, x_boundary = Dirichlet())
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

"""The number of hidden units, which is the number of atoms the fit selects."""
hidden_units(ansatz::NetworkAnsatz) = ansatz.S

"""The number of outputs of the network, which is the number of components of the field."""
noutputs(ansatz::NetworkAnsatz) = ansatz.D

"""The activation of the hidden layer."""
activation_function(ansatz::NetworkAnsatz) = layers(ansatz.chain)[1].σ

"""The number of the network's parameters, in the order `L1.W`, `L1.b`, `L2.W`."""
nparameters(ansatz::NetworkAnsatz) = ansatz.np

"""
    free_parameters(ansatz::NetworkAnsatz)

The indices of the network's parameters that the solver determines: every one of them in `:all`
mode, and the output weights alone in `:last_layer` mode.
"""
function free_parameters(ansatz::NetworkAnsatz)
    first = ansatz.first_free
    return first:(first + ansatz.nfree - 1)
end

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

default_fit(::NetworkAnsatz) = OrthogonalGreedy()

"""
    NetworkFit{ST}

The workspace of the [`OrthogonalGreedy`](@ref) fit, one per network block: the nodes of the slab and
their weights, the dictionary's atoms with its table at those nodes, and the buffers of the greedy
selection — the orthonormal basis of the atoms taken so far with its triangular factor, the
projections of the target on it, the target and its residual, the output weights, the correlations,
the selection, and the mask of the atoms already taken.

Every array is empty until the first fit sizes it, and holds what that fit left for the next one.
"""
mutable struct NetworkFit{ST}
    nodes::Matrix{ST}
    weights::Vector{ST}
    atoms::Matrix{ST}
    table::Matrix{ST}
    basis::Matrix{ST}
    gram::Matrix{ST}
    projection::Matrix{ST}
    correlations::Matrix{ST}
    targets::Matrix{ST}
    residual::Matrix{ST}
    weighted::Matrix{ST}
    coefficients::Matrix{ST}
    work::Vector{ST}
    selected::Vector{Int}
    taken::Vector{Bool}
end

function NetworkFit(::Type{ST}) where {ST}
    return NetworkFit{ST}(zeros(ST, 2, 0), ST[], zeros(ST, 0, 3), zeros(ST, 0, 0),
        zeros(ST, 0, 0), zeros(ST, 0, 0), zeros(ST, 0, 0), zeros(ST, 0, 0), zeros(ST, 0, 0),
        zeros(ST, 0, 0), zeros(ST, 0, 0), zeros(ST, 0, 0), ST[], Int[], Bool[])
end

"""
    size_fit!(workspace, N, S, D, M)

Size the workspace of a greedy fit of `N` nodes, `S` atoms and `D` components over a dictionary of
`M` atoms. Every array is replaced, so a workspace of another size pays for its own tables alone.
"""
function size_fit!(f::NetworkFit{ST}, N::Integer, S::Integer, D::Integer,
        M::Integer) where {ST}
    f.nodes = zeros(ST, 2, N)
    f.weights = zeros(ST, N)
    f.atoms = zeros(ST, M, 3)
    f.table = zeros(ST, M, N)
    f.basis = zeros(ST, N, S)
    f.gram = zeros(ST, S, S)
    f.projection = zeros(ST, S, D)
    f.correlations = zeros(ST, M, D)
    f.targets = zeros(ST, N, D)
    f.residual = zeros(ST, N, D)
    f.weighted = zeros(ST, N, D)
    f.coefficients = zeros(ST, S, D)
    f.work = zeros(ST, N)
    f.selected = zeros(Int, S)
    f.taken = falses(M)
    return f
end

"""
    prepare_fit!(workspace, method::OrthogonalGreedy, tabulation, S, D, activation)

Size the workspace of a greedy fit of `S` atoms and `D` components on the nodes of the slab's own
rule, and build the dictionary with its table there, unless the workspace already holds them.
"""
function prepare_fit!(
        f::NetworkFit{ST}, method::OrthogonalGreedy, tabulation::SlabTabulation,
        S::Integer, D::Integer, activation) where {ST}
    t, x = tabulation.time, tabulation.space
    N = nquadrature(t) * nquadrature(x)
    atoms = greedy_atoms(method, ST)
    m = size(atoms, 1)
    size(f.table) == (m, N) && size(f.coefficients) == (S, D) && return f
    size_fit!(f, N, S, D, m)
    n = 0
    for q in 1:nquadrature(t), r in 1:nquadrature(x)

        n += 1
        f.nodes[1, n] = t.node[q]
        f.nodes[2, n] = x.node[r]
        f.weights[n] = t.weight[q] * x.weight[r]
    end
    copyto!(f.atoms, atoms)
    for n in axes(f.nodes, 2), k in 1:m

        f.table[k, n] = activation(atoms[k, 1] * f.nodes[1, n] +
                                   atoms[k, 2] * f.nodes[2, n] + atoms[k, 3])
    end
    return f
end

"""
    select_direction!(workspace, k, s, N)

Remove from the atom `k` its components along the `s` atoms already taken, into the workspace's
direction buffer, and return the atom's own squared length there. The projections onto the taken
atoms stay in the triangular factor, for the least-squares step that follows.
"""
function select_direction!(f::NetworkFit{ST}, k::Integer, s::Integer, N::Integer) where {ST}
    norm2 = zero(ST)
    for i in 1:N
        f.work[i] = f.table[k, i]
        norm2 += f.weights[i] * abs2(f.work[i])
    end
    for j in 1:s
        dot = zero(ST)
        for i in 1:N
            dot += f.weights[i] * f.basis[i, j] * f.table[k, i]
        end
        f.gram[j, s + 1] = dot
        for i in 1:N
            f.work[i] -= dot * f.basis[i, j]
        end
    end
    gain2 = zero(ST)
    for i in 1:N
        gain2 += f.weights[i] * abs2(f.work[i])
    end
    f.gram[s + 1, s + 1] = gain2
    return norm2
end

"""
    select_atom!(workspace, k, gain, s, N, D)

Take the atom `k`, whose remaining direction has length `gain`, into the orthonormal basis at column
`s`, and read the output weights of every component behind it: its projections on the target, the
least-squares solution through the triangular factor, and the residual the next selection reads.
"""
function select_atom!(f::NetworkFit{ST}, k::Integer, gain::ST, s::Integer, N::Integer,
        D::Integer) where {ST}
    f.selected[s] = k
    f.gram[s, s] = gain
    for i in 1:N
        f.basis[i, s] = f.work[i] / gain
    end
    for d in 1:D
        value = zero(ST)
        for i in 1:N
            value += f.weights[i] * f.basis[i, s] * f.targets[i, d]
        end
        f.projection[s, d] = value
        for i in s:-1:1
            v = f.projection[i, d]
            for j in (i + 1):s
                v -= f.gram[i, j] * f.coefficients[j, d]
            end
            f.coefficients[i, d] = v / f.gram[i, i]
        end
    end
    for d in 1:D, i in 1:N

        v = f.targets[i, d]
        for j in 1:s
            v -= f.basis[i, j] * f.projection[j, d]
        end
        f.residual[i, d] = v
    end
    return f
end

"""
    fit_parameters!(w, workspace, S, D)

Write the fitted atoms into a network's parameter vector, in the order `L1.W`, `L1.b`, `L2.W`: the
two hidden weights and the bias of atom `j`, and the output weight of every component behind it. An
atom the selection passed over leaves its unit zero, so the parameter stays where the caller put it.
"""
function fit_parameters!(w::AbstractVector{ST}, f::NetworkFit{ST}, S::Integer,
        D::Integer) where {ST}
    for j in 1:S
        k = f.selected[j]
        k == 0 && continue
        w[j] = f.atoms[k, 1]
        w[S + j] = f.atoms[k, 2]
        w[2S + j] = f.atoms[k, 3]
        for d in 1:D
            w[3S + (j - 1) * D + d] = f.coefficients[j, d]
        end
    end
    return w
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
derivatives and their parameter derivative — and `input` its two-entry argument.
"""
struct NetworkSlab{ST, T, A <: NetworkAnsatz{T}}
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
    fit::NetworkFit{ST}
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
    return NetworkSlab{ST, T, typeof(a)}(a,
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
        zeros(ST, 2), zeros(ST, 3D), zeros(ST, 3D, a.np), NetworkFit(ST))
end

"""
    ansatz_data_type(ansatz, problem, ST)

The type of [`ansatz_data`](@ref)`(ansatz, problem, ST)`, computed without building the block.
"""
function ansatz_data_type(a::NetworkAnsatz{T}, ::LagrangianPDEProblem,
        ::Type{ST}) where {T, ST}
    return NetworkSlab{ST, T, typeof(a)}
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
scratch.
"""
@inline function evaluate_at!(s::NetworkSlab, t̂, x)
    a = s.ansatz
    s.input[1] = t̂
    s.input[2] = x
    copyto!(s.values, a.field(s.input, s.params))
    copyto!(s.gradient, a.gradient(s.input, s.params))
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
    for d in 1:D, r in axes(sample, 2), q in axes(sample, 1)
        t̂ = convert(ST, s.tabulation.time.node[q])
        evaluate_at!(s, one(ST), s.tabulation.space.node[r])
        # the displacement beyond the previous top, in the normalised time, times the slope there
        sample[q, r, d] = s.values[d] + t̂ * s.values[D + d]
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
# converted to the problem's element type. A `f` that returns a number gives every component that
# number; one that returns a vector gives component `d` its `d`-th entry.
function network_target_values!(sample::AbstractArray{ST, 3}, f, s::NetworkSlab,
        problem::LagrangianPDEProblem, sol) where {ST}
    tq, xq = s.tabulation.time, s.tabulation.space
    t₀ = slab_start(problem, sol)
    h = timestep(problem)
    for q in 1:nquadrature(tq), r in 1:nquadrature(xq)

        value = f(t₀ + h * tq.node[q], xq.node[r], parameters(problem))
        for d in axes(sample, 3)
            sample[q, r, d] = convert(ST, value isa Number ? value : value[d])
        end
    end
    return nothing
end

"""
    apply_guess!(cache, guess, ansatz::NetworkAnsatz, problem, sol)

Fill the solver's vector of a network slab from the method's initial guess.

A continuation carries the previous slab's parameters into the new one, which the block still holds
where the previous step left them; at the first slab there is nothing to carry, and the guess falls
back to `FitTo(InitialCondition())`. A `FitTo` samples its target at the nodes of the ansatz's own
rule, fits the network to it, and takes the free part of the fitted parameters as its unknowns.
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
    fit!(w, fit_method(guess, ansatz), ansatz, sample, c)
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
