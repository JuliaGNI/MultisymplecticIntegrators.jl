"""
    HardConstraintLifting() <: SlabCoupling

The coupling of the hard-constraint formulation: the field of the slab is the ansatz's field
projected onto the slab's data, plus the data.

The projection is the transfinite one, which vanishes at the two ends of `xspan` and at the bottom
of the slab,

```math
P[N] = N - T_1[N] + T_2[N],\\quad
T_1[N] = \\psi_L(x) N(\\hat t, a) + \\psi_R(x) N(\\hat t, b) + (1 - \\hat t) N(0, x),\\quad
T_2[N] = (1 - \\hat t) \\left(\\psi_L(x) N(0, a) + \\psi_R(x) N(0, b)\\right),
```

with `ψ_L(x) = (b - x)/(b - a)` and `ψ_R(x) = (x - a)/(b - a)`, and the lift is its counterpart on
the slab's data,

```math
\\ell = \\psi_L g_a(t) + \\psi_R g_b(t) + (1 - \\hat t) u_b(x)
       - (1 - \\hat t) \\left(\\psi_L g_a(t_0) + \\psi_R g_b(t_0)\\right)
```

at the physical time `t = t_0 + h \\hat t`, with `g_a` and `g_b` the problem's `bcs_function` at the
two ends and `u_b` the bottom datum. The field of the slab is `P[N] + ℓ`, so it is the data at the
three parts of the boundary by construction: `P[N]` is zero there and `ℓ` is the datum.

The corner correction is the term `ψ_L(x) * (1 - t̂)` at the left end and `ψ_R(x) * (1 - t̂)` at the
right. With it `ℓ` interpolates the sides at the slab's start and the bottom along the whole slab;
without it the bottom would carry the sides' values at `t_0` as well, and only the second slab,
whose sides at `t_0` are its predecessor's top, would show it.

The coupling carries no unknown of its own — no multiplier and no coefficient of its own — so
`coupling_unknowns` is zero, [`coupling_term!`](@ref) adds nothing to the residual, and the slab
hands the next one its top row alone. The interior weak form is the whole residual: there is no
surface term and no load at the bottom of the slab.

The side values are the problem's boundary function and the bottom is its initial condition on the
first slab and the previous slab's top afterwards, so the coupling reads no exact solution. A
boundary function that does not meet the initial condition at a corner is a warning, once per
integrator, and never an error.

A [`NetworkAnsatz`](@ref) and a [`TensorBSplineAnsatz`](@ref) are both accepted. For a network the
transform projects the field's derivative with respect to every free parameter together with the
field, because the projection is linear in the parameters. A coefficient ansatz carries no parameter
derivative, and its rows are its own basis functions rather than the projected ones: there the
transform places the data in the field and the residual is a record of what the placement leaves.
"""
struct HardConstraintLifting <: SlabCoupling end

"""The free rows of a network slab are its free parameters."""
function free_rows(::HardConstraintLifting, ansatz::NetworkAnsatz)
    return nfree(ansatz)
end

"""
    free_rows(::HardConstraintLifting, ansatz::TensorBSplineAnsatz)

The free rows of a coefficient slab are all its time functions: the coupling fixes no row of the
coefficient array, and the slab's bottom is imposed on the field instead.
"""
function free_rows(::HardConstraintLifting, ansatz::SpaceTimeAnsatz)
    return ntime_functions(ansatz)
end

"""
    coupling_unknowns(::HardConstraintLifting, ansatz, problem)

The lifting's own unknowns, which are none: the data are imposed by construction and no multiplier
is carried, so the solver's vector is the ansatz's own.
"""
function coupling_unknowns(::HardConstraintLifting, ::SpaceTimeAnsatz,
        ::LagrangianPDEProblem)
    return 0
end

# The coupling fixes no row of the coefficient array.

"""
    HardConstraintLiftingData{ST, C} <: CouplingData

The block of the lifting: the coupling, the slab's start time `start`, the datum's `u` and its
space derivative `w` at the nodes of the rule the residual reads, and one scratch vector `work`.

`start` is the physical time of the slab's bottom, so that a side value is read at `start + h t̂`,
and it moves with the slab; the datum's two arrays move with it too.

`work` is laid out in three parts, the last empty where the ansatz carries no basis of its own:

  - the field of the two corners, `(0, a)` and `(0, b)`, as `[u; ∂_t̂u; ∂ₓu]`, `6D` entries;
  - their derivative with respect to the `nf` free parameters, `6D · nf` entries, entry `(i, k)` of
    the `k`-th parameter at `6D + (k - 1) · 6D + i`;
  - the traces of the basis of a coefficient ansatz at the three boundary parts: the space
    functions at the two ends of `xspan` and their derivative there, `4 · Nx` entries, and the time
    functions at the slab's bottom and their derivative, `2 · Nt` entries.
"""
mutable struct HardConstraintLiftingData{ST, C <: SlabCoupling} <: CouplingData
    coupling::C
    start::ST
    bottom_u::Matrix{ST}
    bottom_w::Matrix{ST}
    work::Vector{ST}
end

# The space nodes of the rule the ansatz's field is evaluated on, which is where the bottom datum
# lives: for a network they are the space nodes of the basis its coupling reads, and for a
# coefficient ansatz those of its own quadrature.
function lifting_space_nodes(ansatz::SpaceTimeAnsatz)
    return quadrature_nodes(quadratures(quadrature(ansatz))[2])
end

# The coupling fixes no row of the coefficient array.
bottom_datum(::HardConstraintLiftingData) = nothing

# The number of entries the block's scratch holds behind the corners: the datum's nodes, and, for a
# coefficient ansatz, the traces of its basis at the three boundary parts.
lifting_tail_count(::NetworkAnsatz, nx::Integer) = nx
function lifting_tail_count(ansatz::TensorBSplineAnsatz, nx::Integer)
    return nx + 4 * nspace_functions(ansatz) + 2 * ntime_functions(ansatz)
end

# The nodes of the datum, and, for a coefficient ansatz, the traces of its basis at the three
# boundary parts — everything the transform reads behind the two corners. They lie in the block's
# scratch from `offset`, the corners' own entries ending where they start.
function lifting_nodes!(data, ansatz::SpaceTimeAnsatz, offset::Integer,
        ::Type{ST}) where {ST}
    for (i, node) in enumerate(lifting_space_nodes(ansatz))
        data.work[offset + i] = convert(ST, node)
    end
    return nothing
end

function lifting_tables!(data, ansatz::SpaceTimeAnsatz, offset::Integer,
        ::Type{ST}) where {ST}
    lifting_nodes!(data, ansatz, offset, ST)
    return nothing
end

function lifting_tables!(data, ansatz::TensorBSplineAnsatz{T}, offset::Integer,
        ::Type{ST}) where {T, ST}
    lifting_nodes!(data, ansatz, offset, ST)
    tb, xb = bases(basis(ansatz))
    nt, nb = nbasis(tb), nbasis(xb)
    xa, xb_end = xspan(ansatz)
    o = offset + length(lifting_space_nodes(ansatz))
    for j in 1:nb
        data.work[o + j] = evaluate(xb, j, convert(T, xa))
        data.work[o + nb + j] = evaluate(xb, j, convert(T, xb_end))
        data.work[o + 2nb + j] = evaluate(xb, j, convert(T, xa), 1)
        data.work[o + 3nb + j] = evaluate(xb, j, convert(T, xb_end), 1)
    end
    o += 4nb
    for i in 1:nt
        data.work[o + i] = evaluate(tb, i, zero(T))
        data.work[o + nt + i] = evaluate(tb, i, zero(T), 1)
    end
    return nothing
end

"""
    lifting_datum(column, nodes, x, left, right)

The datum of one component at `x`, where `left` and `right` are its values at the two ends of
`xspan`.

The datum is stored at `nodes`, the space nodes of the ansatz's own rule, where this returns the
node's own value: the residual and the trace read the datum exactly. Between two nodes the value is
their linear blend, and beyond them — the top of the slab is evaluated on the output grid, whose
points include the two ends — it is `left` or `right`, the datum's value there.
"""
function lifting_datum(column::AbstractVector{ST}, nodes::AbstractVector, x, left,
        right) where {ST}
    n = length(nodes)
    n == 0 && return left
    x < nodes[1] && return left
    x > nodes[n] && return right
    for i in 1:(n - 1)
        x ≤ nodes[i + 1] && return lifting_blend(column, nodes, i, x)
    end
    return column[n]
end

# The linear blend of two neighbouring nodes at `x`, the value of the first where the two coincide.
@inline function lifting_blend(column::AbstractVector{ST}, nodes::AbstractVector,
        i::Integer, x) where {ST}
    span = nodes[i + 1] - nodes[i]
    iszero(span) && return column[i]
    t = (x - nodes[i]) / span
    return column[i] + t * (column[i + 1] - column[i])
end

function coupling_data(c::HardConstraintLifting, ansatz::SpaceTimeAnsatz,
        problem::LagrangianPDEProblem, ::Type{ST}) where {ST}
    nx = length(lifting_space_nodes(ansatz))
    D = components(problem)
    nf = free_rows(c, ansatz)
    tables = 6D + 6D * nf
    data = HardConstraintLiftingData{ST, typeof(c)}(c, convert(ST, timespan(problem)[1]),
        zeros(ST, nx, D), zeros(ST, nx, D),
        zeros(ST, tables + lifting_tail_count(ansatz, nx)))
    lifting_tables!(data, ansatz, tables, ST)
    return data
end

function coupling_data_type(::HardConstraintLifting, ::SpaceTimeAnsatz,
        ::LagrangianPDEProblem, ::Type{ST}) where {ST}
    return HardConstraintLiftingData{ST, HardConstraintLifting}
end

"""
    slab_bottom!(cache, data::HardConstraintLiftingData, problem, ansatz, sol)

Fix the lifting's datum: the slab's start time, and the bottom's `u` and `w` at the space nodes of
the ansatz's rule. The first slab takes them from the problem's initial condition, and every later
slab from the top of the slab before it, which is where the lifting left the field.
"""
function slab_bottom!(c::SpaceTimeGalerkinCache{ST}, data::HardConstraintLiftingData{ST},
        problem::LagrangianPDEProblem, ansatz::SpaceTimeAnsatz, sol) where {ST}
    data.start = convert(ST, slab_start(problem, sol))
    if c.steps == 0
        values = problem.ics_function(collect(lifting_space_nodes(ansatz)),
            parameters(problem))
        data.bottom_u .= values.u
        data.bottom_w .= values.w
    else
        copyto!(data.bottom_u, c.trace_u)
        copyto!(data.bottom_w, c.trace_w)
    end
    return nothing
end

"""
    copy_coupling_data!(cache_data, data::HardConstraintLiftingData)

Copy the datum from the cache of the problem's element type into the cache the solver evaluates in,
which the residual does before every evaluation.
"""
function copy_coupling_data!(c::HardConstraintLiftingData{ST},
        data::HardConstraintLiftingData) where {ST}
    c.start = convert(ST, data.start)
    copyto!(c.bottom_u, data.bottom_u)
    copyto!(c.bottom_w, data.bottom_w)
    return nothing
end

"""
    coupling_term!(b, x, data::HardConstraintLiftingData, cache, ansatz, problem)

The lifting's own part of the residual, which is none: the field carries the data by construction,
so the interior weak form alone is the slab's residual and no surface row is added.
"""
function coupling_term!(b::AbstractVector, x, data::HardConstraintLiftingData,
        c::SpaceTimeGalerkinCache, ansatz::SpaceTimeAnsatz, problem::LagrangianPDEProblem)
    return nothing
end

# The lifting has no unknown of its own to guess.
function coupling_guess!(::HardConstraintLiftingData, cache, ansatz, problem, sol)
    return nothing
end

# And none to read out of the solved slab: what the next slab takes is its top row, which the
# coupling's datum reads from the cache's trace.
function coupling_update!(::HardConstraintLiftingData, x, cache, ansatz, problem)
    return nothing
end

# The lifting keeps nothing in the solution step's internal named tuple.
function coupling_internal(::HardConstraintLifting, ::SpaceTimeAnsatz,
        ::LagrangianPDEProblem)
    return NamedTuple()
end

"""
    check_coupling(problem, ::HardConstraintLifting, ansatz)

Accept both ansaetze, and report a corner at which the problem's boundary function and its initial
condition disagree.

The projection fixes the slab at the two ends in `x` and at its bottom, so the corners `(0, a)` and
`(0, b)` carry two data. A problem whose boundary function is its exact solution meets its initial
condition there exactly; a problem whose data disagree still runs, with the difference carried by
the corner correction, and the warning names the corner and the difference so that such a run is
visible in its log. It is emitted when the integrator is built, so one integrator warns once per
corner.
"""
function check_coupling(problem::LagrangianPDEProblem{T}, ::HardConstraintLifting,
        ansatz::SpaceTimeAnsatz) where {T}
    xa, xb = xspan(problem)
    t₀ = timespan(problem)[1]
    initial = problem.ics_function([xa, xb], parameters(problem))
    sides = problem.bcs_function(t₀, parameters(problem))
    for (corner, values) in ((:left, sides.left), (:right, sides.right))
        r = corner === :left ? 1 : 2
        for d in axes(initial.u, 2)
            u₀ = convert(T, initial.u[r, d])
            g = convert(T, values.u[d])
            scale = max(one(T), abs(u₀), abs(g))
            mismatch = abs(u₀ - g)
            mismatch > 16 * eps(T) * scale &&
                @warn("the lifting's data disagree at the corner x = $corner of component $d: the " *
                      "initial condition is $u₀ where the boundary function is $g, a difference of " *
                      "$mismatch", corner=corner, component=d, mismatch=mismatch)
        end
    end
    return nothing
end

# The field and its parameter derivative at the two corners of the slab, `(0, a)` and `(0, b)`, into
# the block's scratch: the field of the left corner and then of the right one as `[u; ∂_t̂u; ∂ₓu]`,
# and behind them the derivative with respect to every free parameter of the same two triples.
function lifting_corners!(data::HardConstraintLiftingData{ST}, block::NetworkSlab{ST},
        xa, xb) where {ST}
    a = block.ansatz
    D, nf = a.D, a.nfree
    for (corner, point) in ((1, xa), (2, xb))
        evaluate_at!(block, zero(ST), convert(ST, point))
        o = (corner - 1) * 3D
        for i in 1:(3D)
            data.work[o + i] = block.values[i]
            for k in 1:nf
                data.work[6D + (k - 1) * 6D + o + i] = block.gradient[i, a.first_free + k - 1]
            end
        end
    end
    return nothing
end

"""
    transform_field!(u, v, w, data, block::NetworkSlab, t̂, x, h, problem, Ju, Jv, Jw)

The lifting's transform of a network's field: the field, its two derivatives and — where `Ju`, `Jv`
and `Jw` are this evaluation's — its derivative with respect to every free parameter become `P[N] +
ℓ` at the nodes `(t̂[q], x[r])`.

The three parameter-derivative arrays default to the block's own, which are the ones the residual's
evaluation filled. The core also transforms the top of the slab, at one time node and at another set
of space points; that evaluation writes no parameter derivative, and the block's arrays are left
alone there because their sizes are the residual's and not the top's.

The transform reads the ansatz's block and never the ansatz object, so it projects the parameters
the step is solving for: `block.values`, `block.gradient` and the block's surfaces are one
evaluation of `block.params`. It writes the field arrays alone, so the second call of a step — the
one in `update!` — compounds nothing of the slab before it into the field.
"""
function transform_field!(
        u::AbstractArray{ST, 3}, v::AbstractArray{ST, 3}, w::AbstractArray{ST, 3},
        data::HardConstraintLiftingData{ST}, block::NetworkSlab{ST}, t̂, x, h,
        problem::LagrangianPDEProblem,
        Ju = block.Ju, Jv = block.Jv, Jw = block.Jw) where {ST}
    a = block.ansatz
    D, nf = a.D, a.nfree
    xa, xb = xspan(a)
    span = convert(ST, xb - xa)
    hh = convert(ST, h)
    oneST = one(ST)
    params = parameters(problem)
    work = data.work
    # the parameter derivative is projected where the arrays are this evaluation's own
    project_J = size(Ju, 1) == size(u, 1) && size(Ju, 2) == size(u, 2)
    lifting_corners!(data, block, xa, xb)
    # the datum's nodes, which the block tabulated once and an evaluation off them blends
    nodes = view(work, (6D + 6D * nf + 1):(6D + 6D * nf + size(data.bottom_u, 1)))
    # ---- the bottom: the projection's corner-corrected value at t̂ = 0, and the datum it carries
    start_sides = problem.bcs_function(data.start, params)
    for r in axes(u, 2)
        xr = convert(ST, x[r])
        ψL = (convert(ST, xb) - xr) / span
        ψR = (xr - convert(ST, xa)) / span
        evaluate_at!(block, zero(ST), xr)
        for d in 1:D
            # the projection's bottom value and its two derivatives, corner-corrected
            bu = block.values[d] - ψL * work[d] - ψR * work[3D + d]
            bv = block.values[D + d] - ψL * work[D + d] - ψR * work[3D + D + d]
            bw = block.values[2D + d] + work[d] / span - ψL * work[2D + d] -
                 work[3D + d] / span - ψR * work[5D + d]
            # and the datum, less the sides' values at the slab's start
            ga₀ = convert(ST, start_sides.left.u[d])
            gb₀ = convert(ST, start_sides.right.u[d])
            uc = bu - lifting_datum(view(data.bottom_u, :, d), nodes, xr, ga₀, gb₀) +
                 ψL * ga₀ + ψR * gb₀
            wc = bw - lifting_datum(view(data.bottom_w, :, d), nodes, xr, zero(ST),
                zero(ST)) - ga₀ / span + gb₀ / span
            for q in axes(u, 1)
                φ = oneST - convert(ST, t̂[q])
                u[q, r, d] -= φ * uc
                v[q, r, d] += (uc - φ * bv) / hh
                w[q, r, d] -= φ * wc
            end
            project_J || continue
            for k in 1:nf
                p = a.first_free + k - 1
                o = 6D + (k - 1) * 6D
                jbu = block.gradient[d, p] - ψL * work[o + d] - ψR * work[o + 3D + d]
                jbv = block.gradient[D + d, p] - ψL * work[o + D + d] -
                      ψR * work[o + 3D + D + d]
                jbw = block.gradient[2D + d, p] + work[o + d] / span -
                      ψL * work[o + 2D + d] - work[o + 3D + d] / span -
                      ψR * work[o + 5D + d]
                for q in axes(u, 1)
                    φ = oneST - convert(ST, t̂[q])
                    Ju[q, r, d, k] -= φ * jbu
                    Jv[q, r, d, k] += (jbu - φ * jbv) / hh
                    Jw[q, r, d, k] -= φ * jbw
                end
            end
        end
    end
    # ---- the two ends: the projection's term there, and the side values the lift puts beside it
    for (corner, point) in ((1, xa), (2, xb))
        ψ′ = (corner == 1 ? -oneST : oneST) / span
        for q in axes(u, 1)
            tq = t̂[q]
            sides = problem.bcs_function(data.start + hh * convert(ST, tq), params)
            values = corner == 1 ? sides.left : sides.right
            evaluate_at!(block, tq, convert(ST, point))
            for r in axes(u, 2)
                xr = convert(ST, x[r])
                ψ = corner == 1 ? (convert(ST, xb) - xr) / span :
                    (xr - convert(ST, xa)) / span
                for d in 1:D
                    δu = block.values[d] - convert(ST, values.u[d])
                    δv = block.values[D + d] / hh - convert(ST, values.v[d])
                    u[q, r, d] -= ψ * δu
                    v[q, r, d] -= ψ * δv
                    w[q, r, d] -= ψ′ * δu + ψ * block.values[2D + d]
                    project_J || continue
                    for k in 1:nf
                        p = a.first_free + k - 1
                        Ju[q, r, d, k] -= ψ * block.gradient[d, p]
                        Jv[q, r, d, k] -= ψ * block.gradient[D + d, p] / hh
                        Jw[q, r, d, k] -= ψ′ * block.gradient[d, p] +
                                          ψ * block.gradient[2D + d, p]
                    end
                end
            end
        end
    end
    return nothing
end

# The coefficient array reduced at one end of `xspan`: `out[i, d]` is the value and `out_w[i, d]`
# the space derivative of the slab's field there, by time function — what the transform's side terms
# read at every time node.
function lifting_reduction!(out::AbstractMatrix{ST}, out_w::AbstractMatrix{ST},
        coeffs::Array{ST, 3}, values::AbstractVector, slopes::AbstractVector) where {ST}
    nt, nx, D = size(coeffs)
    for d in 1:D, i in 1:nt

        value = zero(ST)
        slope = zero(ST)
        for j in 1:nx
            c = coeffs[i, j, d]
            value += c * values[j]
            slope += c * slopes[j]
        end
        out[i, d] = value
        out_w[i, d] = slope
    end
    return nothing
end

# The field of a coefficient slab at the time node `q` of its own rule, from a reduction at one end:
# the value, its derivative in the normalised time and its space derivative.
function lifting_end_value(tabulation::SlabTabulation, reduction::AbstractMatrix,
        reduction_w::AbstractMatrix, q::Integer, d::Integer)
    u = zero(eltype(reduction))
    v = zero(eltype(reduction))
    w = zero(eltype(reduction))
    t = tabulation.time
    for k in axes(t.index, 2)
        i = t.index[q, k]
        i == 0 && continue
        φ = convert(eltype(reduction), t.value[q, k])
        φ′ = convert(eltype(reduction), t.slope[q, k])
        u += φ * reduction[i, d]
        v += φ′ * reduction[i, d]
        w += φ * reduction_w[i, d]
    end
    return u, v, w
end

# The field of a coefficient slab at the bottom of the slab, at the space node `r`, from the two
# traces of the time functions at zero.
function lifting_bottom_value(tabulation::SlabTabulation, coeffs::Array{ST, 3},
        zero_values::AbstractVector, zero_slopes::AbstractVector, r::Integer,
        d::Integer) where {ST}
    nt = size(coeffs, 1)
    u = zero(ST)
    v = zero(ST)
    w = zero(ST)
    x = tabulation.space
    for i in 1:nt
        φ = convert(ST, zero_values[i])
        φ′ = convert(ST, zero_slopes[i])
        (iszero(φ) && iszero(φ′)) && continue
        value = zero(ST)
        slope = zero(ST)
        for l in axes(x.index, 2)
            j = x.index[r, l]
            j == 0 && continue
            c = coeffs[i, j, d]
            value += c * convert(ST, x.value[r, l])
            slope += c * convert(ST, x.slope[r, l])
        end
        u += φ * value
        v += φ′ * value
        w += φ * slope
    end
    return u, v, w
end

"""
    transform_field!(u, v, w, data, block::TensorBSplineSlab, t̂, x, h, problem)

The lifting's transform of a coefficient slab's field: `P[N] + ℓ` at the nodes `(t̂[q], x[r])`, for
the field and its two derivatives.

A coefficient ansatz carries no parameter derivative, and its rows are its own basis functions
rather than the projected ones; the transform therefore places the data in the field and the
residual of such a slab records what the placement leaves.
"""
function transform_field!(
        u::AbstractArray{ST, 3}, v::AbstractArray{ST, 3}, w::AbstractArray{ST, 3},
        data::HardConstraintLiftingData{ST}, block::TensorBSplineSlab{ST}, t̂, x, h,
        problem::LagrangianPDEProblem) where {ST}
    coeffs = block.coeffs
    nt, nb, D = size(coeffs)
    tabulation = block.tabulation
    xa, xb = xspan(problem)
    span = convert(ST, xb - xa)
    hh = convert(ST, h)
    oneST = one(ST)
    params = parameters(problem)
    work = data.work
    nn = size(data.bottom_u, 1)
    # the datum's nodes, and behind them the basis traces at the three boundary parts, both of which
    # the integrator tabulated once
    o = 6D + 6D * nt
    nodes = view(work, (o + 1):(o + nn))
    o += nn
    ψa = view(work, (o + 1):(o + nb))
    ψb = view(work, (o + nb + 1):(o + 2nb))
    ψa′ = view(work, (o + 2nb + 1):(o + 3nb))
    ψb′ = view(work, (o + 3nb + 1):(o + 4nb))
    o += 4nb
    φ₀ = view(work, (o + 1):(o + nt))
    φ₀′ = view(work, (o + nt + 1):(o + 2nt))
    # the coefficient array reduced at the two ends, and at zero in time
    a = zeros(ST, nt, D)
    aw = zeros(ST, nt, D)
    b = zeros(ST, nt, D)
    bw = zeros(ST, nt, D)
    lifting_reduction!(a, aw, coeffs, ψa, ψa′)
    lifting_reduction!(b, bw, coeffs, ψb, ψb′)
    ca_u = zeros(ST, D)
    ca_v = zeros(ST, D)
    ca_w = zeros(ST, D)
    cb_u = zeros(ST, D)
    cb_v = zeros(ST, D)
    cb_w = zeros(ST, D)
    for d in 1:D
        ua = uv = uw = zero(ST)
        ub = vb = wb = zero(ST)
        for i in 1:nt
            ua += φ₀[i] * a[i, d]
            uv += φ₀′[i] * a[i, d]
            uw += φ₀[i] * aw[i, d]
            ub += φ₀[i] * b[i, d]
            vb += φ₀′[i] * b[i, d]
            wb += φ₀[i] * bw[i, d]
        end
        ca_u[d], ca_v[d], ca_w[d] = ua, uv, uw
        cb_u[d], cb_v[d], cb_w[d] = ub, vb, wb
    end
    # the field at the bottom of the slab and its two derivatives, at the nodes of the ansatz's own
    # rule, corner-corrected — an evaluation off those nodes blends them, as the datum's is blended
    cu = zeros(ST, nn, D)
    cv = zeros(ST, nn, D)
    cw = zeros(ST, nn, D)
    for r in 1:nn
        xr = convert(ST, nodes[r])
        ψL = (convert(ST, xb) - xr) / span
        ψR = (xr - convert(ST, xa)) / span
        for d in 1:D
            cu[r, d], cv[r, d], cw[r, d] = lifting_bottom_value(
                tabulation, coeffs, φ₀, φ₀′,
                r, d)
            cu[r, d] -= ψL * ca_u[d] + ψR * cb_u[d]
            cv[r, d] -= ψL * ca_v[d] + ψR * cb_v[d]
            cw[r, d] += ca_u[d] / span - ψL * ca_w[d] - cb_u[d] / span - ψR * cb_w[d]
        end
    end
    # ---- the bottom: the projection's corner-corrected value at t̂ = 0, and the datum it carries
    start_sides = problem.bcs_function(data.start, params)
    for r in axes(u, 2)
        xr = convert(ST, x[r])
        ψL = (convert(ST, xb) - xr) / span
        ψR = (xr - convert(ST, xa)) / span
        for d in 1:D
            ga₀ = convert(ST, start_sides.left.u[d])
            gb₀ = convert(ST, start_sides.right.u[d])
            uc = lifting_datum(view(cu, :, d), nodes, xr, zero(ST), zero(ST)) -
                 lifting_datum(view(data.bottom_u, :, d), nodes, xr, ga₀, gb₀) + ψL * ga₀ +
                 ψR * gb₀
            wc = lifting_datum(view(cw, :, d), nodes, xr, zero(ST), zero(ST)) -
                 lifting_datum(view(data.bottom_w, :, d), nodes, xr, zero(ST), zero(ST)) -
                 ga₀ / span + gb₀ / span
            for q in axes(u, 1)
                φ = oneST - convert(ST, t̂[q])
                u[q, r, d] -= φ * uc
                v[q, r, d] += (uc -
                               φ * lifting_datum(
                    view(cv, :, d), nodes, xr, zero(ST), zero(ST))) /
                              hh
                w[q, r, d] -= φ * wc
            end
        end
    end
    # ---- the two ends
    for (corner, point) in ((1, xa), (2, xb))
        ψ′ = (corner == 1 ? -oneST : oneST) / span
        for q in axes(u, 1)
            tq = t̂[q]
            sides = problem.bcs_function(data.start + hh * convert(ST, tq), params)
            values = corner == 1 ? sides.left : sides.right
            for r in axes(u, 2)
                xr = convert(ST, x[r])
                ψ = corner == 1 ? (convert(ST, xb) - xr) / span :
                    (xr - convert(ST, xa)) / span
                for d in 1:D
                    eu, ev, ew = lifting_end_value(tabulation,
                        corner == 1 ? a : b, corner == 1 ? aw : bw, q, d)
                    δu = eu - convert(ST, values.u[d])
                    δv = ev / hh - convert(ST, values.v[d])
                    u[q, r, d] -= ψ * δu
                    v[q, r, d] -= ψ * δv
                    w[q, r, d] -= ψ′ * δu + ψ * ew
                end
            end
        end
    end
    return nothing
end
