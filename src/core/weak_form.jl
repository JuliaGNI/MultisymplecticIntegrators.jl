"""
    AxisTabulation{T}

The assembly tables of one axis of the tensor ansatz: at every point of the composite Gauss rule,
the indices of the basis functions that are nonzero there and the values of those functions and of
their first derivative, together with the nodes and the weights.

The tables are dense, `q × width`: a basis function of degree `p` is nonzero on `p + 1` cells, so
the support at one point is `width` functions wide (`p + 1` away from an end, wider for a recombined
basis, whose functions may reach further). An entry whose index is zero is padding and carries no
term.
"""
struct AxisTabulation{T}
    nbasis::Int
    index::Matrix{Int}
    value::Matrix{T}
    slope::Matrix{T}
    node::Vector{T}
    weight::Vector{T}
end

function AxisTabulation(axis::SplineQuadrature{T}) where {T}
    Φ0 = basis_values(axis, 0)
    Φ1 = basis_values(axis, 1)
    node = quadrature_nodes(axis)
    weight = quadrature_weights(axis)
    q = length(node)
    width = maximum(length(nzrange(Φ0, k)) for k in 1:q)
    n = size(Φ0, 1)
    index = zeros(Int, q, width)
    value = zeros(T, q, width)
    slope = zeros(T, q, width)
    for k in 1:q
        for (l, entry) in enumerate(nzrange(Φ0, k))
            index[k, l] = rowvals(Φ0)[entry]
            value[k, l] = nonzeros(Φ0)[entry]
        end
        for (l, entry) in enumerate(nzrange(Φ1, k))
            slope[k, l] = nonzeros(Φ1)[entry]
        end
    end
    return AxisTabulation{T}(n, index, value, slope, node, weight)
end

"""The number of Gauss points of the axis."""
nquadrature(axis::AxisTabulation) = size(axis.index, 1)

"""
    SlabTabulation{T}

The assembly tables of a tensor B-spline ansatz: one [`AxisTabulation`](@ref) per direction.
"""
struct SlabTabulation{T}
    time::AxisTabulation{T}
    space::AxisTabulation{T}
end

function SlabTabulation(ansatz::TensorBSplineAnsatz{T}) where {T}
    rules = quadratures(quadrature(ansatz))
    return SlabTabulation{T}(AxisTabulation(rules[1]), AxisTabulation(rules[2]))
end

"""
    slab_values!(u, v, w, coeffs, tabulation, h)

The field and its two derivatives at every point of the quadrature grid, from the coefficient array
`coeffs` of size `(Nt, Nx, D)`, laid out component-major.

`v` is the derivative in physical time, `∂ₜu = ∂_{t̂}u / h`, and `w` the derivative in space.
"""
function slab_values!(
        u::AbstractArray{ST, 3}, v::AbstractArray{ST, 3}, w::AbstractArray{ST, 3},
        coeffs::AbstractArray{ST, 3}, tabulation::SlabTabulation, h) where {ST}
    t = tabulation.time
    x = tabulation.space
    hh = convert(ST, h)
    D = size(coeffs, 3)
    fill!(u, zero(ST))
    fill!(v, zero(ST))
    fill!(w, zero(ST))
    for d in 1:D
        for q in 1:nquadrature(t), r in 1:nquadrature(x)

            value = zero(ST)
            velocity = zero(ST)
            gradient = zero(ST)
            for k in axes(t.index, 2)
                ik = t.index[q, k]
                ik == 0 && continue
                φ = t.value[q, k]
                φ′ = t.slope[q, k]
                for l in axes(x.index, 2)
                    jl = x.index[r, l]
                    jl == 0 && continue
                    ψ = x.value[r, l]
                    ψ′ = x.slope[r, l]
                    c = coeffs[ik, jl, d]
                    value += c * φ * ψ
                    velocity += c * φ′ * ψ
                    gradient += c * φ * ψ′
                end
            end
            u[q, r, d] = value
            v[q, r, d] = velocity / hh
            w[q, r, d] = gradient
        end
    end
    return nothing
end

"""
    density_derivatives!(∂u, ∂v, ∂w, density, u, v, w, params)

`∂L/∂u`, `∂L/∂v` and `∂L/∂w` at every point of the quadrature grid, from the sampled field.

A `LagrangianDensity` takes one component at a time and is evaluated that way, without allocating.
A density whose components couple — the vector density of `VectorWave` — takes all of them at once,
and is given the `D` components of each point as a vector.
"""
function density_derivatives!(∂u::AbstractArray{ST, 3}, ∂v::AbstractArray{ST, 3},
        ∂w::AbstractArray{ST, 3}, density::LagrangianDensity, u::AbstractArray{ST, 3},
        v::AbstractArray{ST, 3}, w::AbstractArray{ST, 3}, params) where {ST}
    for d in axes(u, 3), r in axes(u, 2), q in axes(u, 1)
        ∂u[q, r, d] = derivative_u(density, u[q, r, d], v[q, r, d], w[q, r, d], params)
        ∂v[q, r, d] = derivative_v(density, u[q, r, d], v[q, r, d], w[q, r, d], params)
        ∂w[q, r, d] = derivative_w(density, u[q, r, d], v[q, r, d], w[q, r, d], params)
    end
    return nothing
end

function density_derivatives!(∂u::AbstractArray{ST, 3}, ∂v::AbstractArray{ST, 3},
        ∂w::AbstractArray{ST, 3}, density, u::AbstractArray{ST, 3},
        v::AbstractArray{ST, 3}, w::AbstractArray{ST, 3}, params) where {ST}
    for r in axes(u, 2), q in axes(u, 1)

        d = derivatives(
            density, view(u, q, r, :), view(v, q, r, :), view(w, q, r, :), params)
        for c in axes(u, 3)
            ∂u[q, r, c] = d.u[c]
            ∂v[q, r, c] = d.v[c]
            ∂w[q, r, c] = d.w[c]
        end
    end
    return nothing
end

"""
    interior_weak_form!(residual, ∂u, ∂v, ∂w, tabulation, h)

The interior weak Euler–Lagrange form of one slab, from the derivatives of the density at the
quadrature points.

For a test function `φ` it reads

```math
\\int_0^1 \\!\\! \\int_{x_{\\min}}^{x_{\\max}}
\\Bigl( h \\, \\frac{\\partial L}{\\partial u} \\, \\phi
     + \\frac{\\partial L}{\\partial v} \\, \\partial_{\\hat t} \\phi
     + h \\, \\frac{\\partial L}{\\partial w} \\, \\partial_x \\phi \\Bigr) \\, dx \\, d\\hat t ,
```

which is the variation of the action `\\int \\int L \\, dt \\, dx` with the derivatives left on the
test function, `dt = h \\, d\\hat t`. The one factor of `h` that the `v` term does not carry is the
`1 / h` of `∂ₜu = ∂_{t̂}u / h`, so the three terms are the three of `δS` with no factor dropped. The
residual is that form for every basis function, in the coefficient layout of the slab.
"""
function interior_weak_form!(residual::AbstractArray{ST, 3}, ∂u::AbstractArray{ST, 3},
        ∂v::AbstractArray{ST, 3}, ∂w::AbstractArray{ST, 3}, tabulation::SlabTabulation,
        h) where {ST}
    t = tabulation.time
    x = tabulation.space
    hh = convert(ST, h)
    fill!(residual, zero(ST))
    for d in axes(residual, 3), q in 1:nquadrature(t), r in 1:nquadrature(x)
        ω = t.weight[q] * x.weight[r]
        ζu = hh * ω * ∂u[q, r, d]
        ζv = ω * ∂v[q, r, d]
        ζw = hh * ω * ∂w[q, r, d]
        for k in axes(t.index, 2)
            ik = t.index[q, k]
            ik == 0 && continue
            φ = t.value[q, k]
            φ′ = t.slope[q, k]
            for l in axes(x.index, 2)
                jl = x.index[r, l]
                jl == 0 && continue
                ψ = x.value[r, l]
                ψ′ = x.slope[r, l]
                residual[ik, jl, d] += ζu * φ * ψ + ζv * φ′ * ψ + ζw * φ * ψ′
            end
        end
    end
    return nothing
end

"""
    boundary_form!(load, values, tabulation)

The weak form of a boundary trace: the load `∫ values(x) φ_j(x) dx` of a trace given at the nodes of
the space quadrature, on the functions of the space basis.

It is the top-row weak form that a [`CoefficientRestriction`](@ref) carries into the next slab, and
the initial momentum of the first slab, which is why both are the same construction.
"""
function boundary_form!(load::AbstractMatrix{ST}, values::AbstractMatrix{ST},
        tabulation::SlabTabulation) where {ST}
    x = tabulation.space
    fill!(load, zero(ST))
    for d in axes(values, 2), r in 1:nquadrature(x)

        ω = x.weight[r] * values[r, d]
        for l in axes(x.index, 2)
            jl = x.index[r, l]
            jl == 0 && continue
            load[jl, d] += ω * x.value[r, l]
        end
    end
    return nothing
end
