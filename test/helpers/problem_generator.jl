using GeometricBase: timesteps
using GeometricIntegratorsBase: GeometricIntegrator, cache, current, reset!, solutionstep
using GeometricSolutions: GeometricSolution
using MultisymplecticIntegrators
using Random: AbstractRNG
using SimpleSplines: Dirichlet, Free
using Symbolics: Symbolics

"""The seed of the property suite's generated problems."""
const SEED = 20261031

"""
    GeneratedProblem

One problem of the property suite and the ansatz it is integrated with.

`kind` is `:general`, `:noether` or `:symmetric`. `potential` is the degree of the potential `V`:
2 for a linear field equation, 3 or 4 for a nonlinear one. `degree`, `ncells` and `n` are the
ansatz's per `(time, space)`, and `x_boundary` is the space basis that `CoefficientRestriction`
takes; the two multiplier couplings take the basis they need.
"""
struct GeneratedProblem{P, B}
    problem::P
    kind::Symbol
    potential::Int
    degree::Tuple{Int, Int}
    ncells::Tuple{Int, Int}
    n::Tuple{Int, Int}
    x_boundary::B
end

"""The three couplings the suite runs every property under."""
const COUPLINGS = (CoefficientRestriction(), WeakInitialCondition(),
    WeakInitialAndBoundaryConditions())

"""
    generated_density(params)

The scalar density of every generated problem, `L = (v² − c²w²)/2 − V(u)` with the potential
`V(u) = a₁u + a₂u²/2 + a₃u³/3 + a₄u⁴/4`, whose coefficients are parameters. One symbolic expression
serves every problem, so that one density type compiles the residual once.
"""
function generated_density(params)
    u, v, w, c, a₁, a₂, a₃, a₄ = Symbolics.variable.((:u, :v, :w, :c, :a1, :a2, :a3, :a4))
    L = (v^2 - c^2 * w^2) / 2 - (a₁ * u + a₂ * u^2 / 2 + a₃ * u^3 / 3 + a₄ * u^4 / 4)
    return LagrangianDensity(L, u, v, w; params = params)
end

# A draw in `[lo, hi]`, at element type `T`.
function uniform(rng::AbstractRNG, ::Type{T}, lo, hi) where {T}
    T(lo) + (T(hi) - T(lo)) * rand(rng, T)
end

# A draw in `[lo, hi]` with a random sign.
function signed(rng::AbstractRNG, ::Type{T}, lo, hi) where {T}
    rand(rng, Bool) ? uniform(rng, T, lo, hi) :
    -uniform(rng, T, lo, hi)
end

"""
    FieldData{T}

The initial data of one component as a sum of modes in `s = (x − a)/(b − a)`:
`u₀ = Σₖ Aₖ m(kπs + φₖ)` and `v₀ = Σₖ Bₖ m(kπs + χₖ)`, with `m = sin`, or `m = cos` and the
frequencies `2kπ` for the data symmetric about the middle of the span. `A₀` and `B₀` are constant
terms.
"""
struct FieldData{T}
    A₀::T
    A::Vector{T}
    φ::Vector{T}
    B₀::T
    B::Vector{T}
    χ::Vector{T}
    symmetric::Bool
end

function field_values(f::FieldData{T}, x::T, a::T, b::T) where {T}
    s = (x - a) / (b - a)
    u = f.A₀
    v = f.B₀
    w = zero(T)
    for k in eachindex(f.A)
        ω = (f.symmetric ? 2k : k) * T(π)
        m, m′ = f.symmetric ? (cos(ω * s + f.φ[k]), -sin(ω * s + f.φ[k])) :
                (sin(ω * s + f.φ[k]), cos(ω * s + f.φ[k]))
        u += f.A[k] * m
        w += f.A[k] * ω / (b - a) * m′
        v += f.B[k] * (f.symmetric ? cos(ω * s + f.χ[k]) : sin(ω * s + f.χ[k]))
    end
    return u, v, w
end

"""
    SideData{T}

The side data of one component at one end of the span, `g(t) = α + β sin(ωt + φ)`.
"""
struct SideData{T}
    α::T
    β::T
    ω::T
    φ::T
end

side_value(g::SideData, t) = g.α + g.β * sin(g.ω * t + g.φ)

"""
    data_functions(fields, left, right, xspan)

The problem's `ics_function` and `bcs_function` from the initial data of each component and the side
data of each end and component, in the layouts `LagrangianPDEProblem` documents: each initial datum
component-major, and the side trace of component `d` at time `t₀` in entry `2(d − 1) + 1`.
"""
function data_functions(fields::Vector{FieldData{T}}, left::Vector{SideData{T}},
        right::Vector{SideData{T}}, xspan::NTuple{2, T}) where {T}
    a, b = xspan
    function ics(x, params)
        values = [field_values(f, xi, a, b) for xi in x, f in fields]
        return (u = vec(map(first, values)), v = vec(map(v -> v[2], values)),
            w = vec(map(last, values)))
    end
    function bcs(times, params)
        side(g) = vec([side_value(gd, t) for t in times, gd in g])
        return (left = (u = side(left),), right = (u = side(right),))
    end
    return ics, bcs
end

"""
    generate_problems(T; rng::AbstractRNG)

The problems of the property suite, at element type `T`, drawn from `rng` alone: 24 entries, of
which 8 are Noether problems, 4 are symmetric problems and 12 are general ones.

- **Noether**: `D = 2`, the rotation-invariant `V(|u|²) = a₂|u|²/2 + β|u|⁴/4` through
  `VectorWave.VectorDensity`, zero side data, and initial data that vanish at both ends of the span
  with `u²` not proportional to `u¹`.
- **Symmetric**: data symmetric under `x ↦ a + b − x`, not constant and not zero at the ends, and
  side data equal on both sides; `D = 1` or `D = 2` with the scalar density on each component.
- **General**: `D = 1` or `D = 2`, the scalar density, side data that vary in time and differ per
  side and per component.

The potentials are of degree 2 in ten problems and of degree 3 or 4 in fourteen. Every initial
velocity is non-zero in every component. Every odd entry starts at a random `timespan[1] ≠ 0`, and
every fifth has a time rule longer than the space rule. Spans, timesteps and wave speeds are random,
so none of them is dyadic.
"""
function generate_problems(::Type{T}; rng::AbstractRNG) where {T}
    density = generated_density((c = one(T), a1 = zero(T), a2 = zero(T), a3 = zero(T),
        a4 = zero(T)))
    kinds = vcat(fill(:noether, 8), fill(:symmetric, 4), fill(:general, 12))
    potentials = vcat(fill(4, 8), [2, 2, 3, 4], [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4])
    dims = vcat(fill(2, 8), [1, 2, 1, 2], [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    entries = GeneratedProblem[]
    for (i, (kind, potential, D)) in enumerate(zip(kinds, potentials, dims))
        push!(entries,
            generate_problem(T, rng, density, i, kind, potential, D))
    end
    return entries
end

function generate_problem(::Type{T}, rng::AbstractRNG, density, i::Integer, kind::Symbol,
        potential::Integer, D::Integer) where {T}
    # the span: random ends and length, never dyadic
    a = uniform(rng, T, -0.6, 0.4)
    m = rand(rng, 8:12)
    b = a + uniform(rng, T, 0.6, 1.3)
    # a step of `m` cells whose range `a:xstep:b`, the problem's output grid, reaches `b`: a step
    # that rounds up stops the range one point short
    xstep = (b - a) / m
    while abs(last(a:xstep:b) - b) > xstep / 4
        xstep = prevfloat(xstep)
    end
    # the time grid: three steps of a random timestep, from zero or from a random time
    h = uniform(rng, T, 0.02, 0.09)
    t₀ = isodd(i) ? uniform(rng, T, -0.7, 0.9) : zero(T)
    timespan = (t₀, t₀ + 3h)
    # the ansatz: a time rule longer than the space rule for every fifth entry
    if i % 5 == 0
        degree = (3, 2)
        ncells = (3, 3)
        n = (5, 3)
    else
        degree = (rand(rng, 1:3), rand(rng, 2:3))
        ncells = (rand(rng, 1:3), rand(rng, 4:7))
        n = (degree[1] + rand(rng, 1:2), degree[2] + rand(rng, 1:2))
    end
    x_boundary = isodd(i) ? Dirichlet() : Free()
    # the potential's coefficients, and the quartic coupling of the vector density
    c = uniform(rng, T, 0.3, 1.2)
    a₂ = uniform(rng, T, 0.2, 1.5)
    if kind === :noether
        a₁, a₃, a₄ = zero(T), zero(T), zero(T)
        β = uniform(rng, T, 0.3, 1.0)
    else
        a₁ = signed(rng, T, 0.1, 0.8)
        a₃ = potential == 3 ? signed(rng, T, 0.3, 1.0) : zero(T)
        a₄ = potential == 4 ? uniform(rng, T, 0.3, 1.0) : zero(T)
        β = zero(T)
    end
    params = (c = T(c), a1 = T(a₁), a2 = T(a₂), a3 = T(a₃), a4 = T(a₄))
    # the initial data of each component
    fields = [initial_field(T, rng, kind, d) for d in 1:D]
    # the side data of each end and component
    left, right = side_data(T, rng, kind, D)
    ics, bcs = data_functions(fields, left, right, (a, b))
    L = kind === :noether ? VectorWave.VectorDensity(density, D, T(β)) : density
    problem = LagrangianPDEProblem(L; D = D, ics_function = ics, bcs_function = bcs,
        timespan = timespan, timestep = h, xspan = (a, b), xstep = xstep, params = params)
    return GeneratedProblem(problem, kind, potential, degree, ncells, n, x_boundary)
end

function initial_field(::Type{T}, rng::AbstractRNG, kind::Symbol, d::Integer) where {T}
    if kind === :noether
        # zero at both ends; the two components take different modes, so that neither is a
        # multiple of the other and the rotation charge is not zero
        A = d == 1 ? [signed(rng, T, 0.5, 1.0), zero(T)] :
            [zero(T), signed(rng, T, 0.5, 1.0)]
        B = d == 1 ? [zero(T), signed(rng, T, 0.5, 1.0)] :
            [signed(rng, T, 0.5, 1.0), zero(T)]
        return FieldData(zero(T), A, zeros(T, 2), zero(T), B, zeros(T, 2), false)
    elseif kind === :symmetric
        # cosines of `2kπs` with no phase: symmetric about the middle, and non-zero at the ends
        A = [signed(rng, T, 0.3, 1.0), signed(rng, T, 0.1, 0.5)]
        A₀ = sum(A) > 0 ? uniform(rng, T, 0.2, 0.6) : -uniform(rng, T, 0.2, 0.6)
        B₀ = signed(rng, T, 0.3, 0.8)
        B = [signed(rng, T, 0.1, 0.5), zero(T)]
        return FieldData(A₀, A, zeros(T, 2), B₀, B, zeros(T, 2), true)
    else
        A = [signed(rng, T, 0.2, 1.0), signed(rng, T, 0.1, 0.5)]
        φ = [uniform(rng, T, 0, 2π), uniform(rng, T, 0, 2π)]
        B = [signed(rng, T, 0.1, 0.5), signed(rng, T, 0.1, 0.5)]
        χ = [uniform(rng, T, 0, 2π), uniform(rng, T, 0, 2π)]
        return FieldData(signed(rng, T, 0.1, 0.5), A, φ, signed(rng, T, 0.3, 0.8), B, χ,
            false)
    end
end

function side_data(::Type{T}, rng::AbstractRNG, kind::Symbol, D::Integer) where {T}
    draw() = SideData(signed(rng, T, 0.2, 0.8), signed(rng, T, 0.2, 0.6),
        uniform(rng, T, 1, 6), uniform(rng, T, 0, 2π))
    if kind === :noether
        zero_side = SideData(zero(T), zero(T), zero(T), zero(T))
        return fill(zero_side, D), fill(zero_side, D)
    elseif kind === :symmetric
        left = [draw() for _ in 1:D]
        return left, copy(left)
    else
        return [draw() for _ in 1:D], [draw() for _ in 1:D]
    end
end

"""
    entry_method(entry, coupling)

The method of a generated problem under one coupling: the entry's tensor B-spline ansatz with the
space basis the coupling takes — the entry's own under `CoefficientRestriction`, `Dirichlet()` under
`WeakInitialCondition` and `Free()` under `WeakInitialAndBoundaryConditions`.
"""
function entry_method(entry::GeneratedProblem, coupling)
    x_boundary = coupling isa CoefficientRestriction ? entry.x_boundary :
                 coupling isa WeakInitialCondition ? Dirichlet() : Free()
    ansatz = TensorBSplineAnsatz(degree = entry.degree, ncells = entry.ncells,
        xspan = xspan(entry.problem), n = entry.n, x_boundary = x_boundary)
    return SpaceTimeGalerkin(ansatz, coupling)
end

"""
    first_slab(entry, coupling; kwargs...)

The integrator of an entry under one coupling, with the datum of its first slab fixed by the
coupling's `slab_bottom!`, and the solution and solution step a run of it drives. The keyword
arguments go to the integrator.
"""
function first_slab(entry::GeneratedProblem, coupling; kwargs...)
    problem = entry.problem
    method = entry_method(entry, coupling)
    int = GeometricIntegrator(problem, method; kwargs...)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    slab = (; problem, method, int, solution, solstep)
    next_slab!(slab, 1)
    return slab
end

"""
    next_slab!(slab, n)

Move the solution step of `slab`, as [`first_slab`](@ref) returns it, to step `n`, and fix the datum
of that slab, as a run does before it solves the slab.
"""
function next_slab!(slab, n::Integer)
    reset!(slab.solstep, timesteps(slab.solution)[n])
    MultisymplecticIntegrators.slab_bottom!(cache(slab.int), coupling(slab.method),
        slab.problem, ansatz(slab.method), current(slab.solstep))
    return nothing
end
