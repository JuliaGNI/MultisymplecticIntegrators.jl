using GeometricBase: datatype, ntime, parameters, timespan, timestep
using GeometricSolutions
using MultisymplecticIntegrators
using Random
using Test

include("../../helpers/testsetup.jl")
include("../../helpers/problem_generator.jl")

const ORACLE = joinpath(@__DIR__, "..", "..", "helpers", "action_oracle.jl")
const SOURCE = joinpath(@__DIR__, "..", "..", "..", "src")

"""The names of the package's assembly that the action oracle must not use."""
const FORBIDDEN = [
    "residual!", "density_derivatives!", "interior_weak_form!", "slab_values!",
    "SlabTabulation", "AxisTabulation", "SlabEvaluation", "trace!", "assign!", "momentum_load!",
    "flux_load!", "coupling_term!", "boundary_form!", "derivative_u", "derivative_v",
    "derivative_w", "derivatives", "free_index", "scatter!", "multiplier_table", "least_squares",
    "nodal_fit"]

"""
A use of `name`: a call `name(` with no identifier character or `.` before it, or a qualified
reference `.name` with no identifier character or `!` after it. Prose in a comment, such as "the
derivatives of L", is no use.
"""
function use_pattern(name::AbstractString)
    n = replace(name, "!" => "\\!")
    return Regex("(^|[^[:alnum:]_.])" * n * "\\(|\\." * n * "([^[:alnum:]_!]|\$)", "m")
end

"""Every value of a generated problem that the suite reads, to compare two generated sets."""
function fingerprint(entry::GeneratedProblem)
    p = entry.problem
    x = xgrid(p)
    t = timespan(p)[1]
    L = p.density
    return (entry.kind, entry.potential, entry.degree, entry.ncells, entry.n,
        typeof(entry.x_boundary), components(p), parameters(p), timespan(p), timestep(p),
        xspan(p), xstep(p), p.ics_function(x, parameters(p)),
        p.bcs_function((t, t), parameters(p)), L isa VectorWave.VectorDensity ? L.β :
                                               nothing)
end

# the total node counts of the time and the space rule of an entry
rule_lengths(e::GeneratedProblem) = (e.n[1] * e.ncells[1], e.n[2] * e.ncells[2])

# the initial data of component `d` on the points `x`
function component(entry::GeneratedProblem, x, d)
    values = entry.problem.ics_function(x, parameters(entry.problem))
    n = length(x)
    pick(f) = f[((d - 1) * n + 1):(d * n)]
    return pick(values.u), pick(values.v), pick(values.w)
end

# the side data of component `d` at time `t`, `(g_a, g_b)`
function sides(entry::GeneratedProblem, t, d)
    values = entry.problem.bcs_function((t, t), parameters(entry.problem))
    return values.left.u[2 * (d - 1) + 1], values.right.u[2 * (d - 1) + 1]
end

dyadic(x) = isinteger(x * 2^12)

@testset "A1a the oracle reads no assembly of the package" begin
    source = read(ORACLE, String)
    package = join(
        (read(joinpath(root, f), String)
        for (root, _, files) in walkdir(SOURCE) for f in files
        if endswith(f, ".jl")),
        "\n")
    for name in FORBIDDEN
        # the pattern finds the name where the package uses it, and not in the oracle
        @test occursin(use_pattern(name), package)
        @test !occursin(use_pattern(name), source)
    end
    @test occursin("ForwardDiff.", source)
end

@testset "A2a the controls of the generated set, $T" for T in (Float64, Float32)
    first_set = generate_problems(T; rng = Xoshiro(SEED))
    second_set = generate_problems(T; rng = Xoshiro(SEED))
    rand(100)
    third_set = generate_problems(T; rng = Xoshiro(SEED))
    @test map(fingerprint, first_set) == map(fingerprint, second_set)
    @test map(fingerprint, first_set) == map(fingerprint, third_set)

    set = first_set
    floor = T(1 // 100)
    @test length(set) ≥ 20
    @test Set(components(e.problem) for e in set) == Set([1, 2])
    @test Set(e.potential for e in set) == Set([2, 3, 4])
    vector = [e for e in set if e.problem.density isa VectorWave.VectorDensity]
    @test any(e -> components(e.problem) == 2 && e.problem.density isa LagrangianDensity, set)
    @test !isempty(vector)
    for e in set
        p = parameters(e.problem)
        β = e.problem.density isa VectorWave.VectorDensity ? e.problem.density.β : zero(T)
        # the potential's degree is the highest coefficient that is not zero
        if e.potential == 2
            @test iszero(p.a3) && iszero(p.a4) && iszero(β) && !iszero(p.a2)
        elseif e.potential == 3
            @test !iszero(p.a3) && iszero(p.a4) && iszero(β)
        else
            @test !iszero(p.a4) || !iszero(β)
        end
        # every parameter and datum has the element type of the set
        assert_no_upcast(p, T)
        assert_no_upcast(β, T)
        assert_no_upcast(timespan(e.problem)[1], T)
        assert_no_upcast(timestep(e.problem), T)
        assert_no_upcast(xspan(e.problem)[1], T)
        assert_no_upcast(e.problem.ics_function(xgrid(e.problem), p), T)
        t = timespan(e.problem)[1]
        assert_no_upcast(e.problem.bcs_function((t, t), p).left.u, T)
        # a timestep that is not one, and spans whose ends are not dyadic
        @test timestep(e.problem) != one(T)
        @test !dyadic(timestep(e.problem))
        @test !dyadic(xspan(e.problem)[1]) && !dyadic(xspan(e.problem)[2])
        @test e.degree[1] in 1:3 && e.degree[2] in 2:3
        @test ntime(GeometricSolution(e.problem)) ≥ 2
        # a non-zero initial velocity in every component
        for d in 1:components(e.problem)
            _, v, _ = component(e, xgrid(e.problem), d)
            @test maximum(abs, v) ≥ floor
        end
    end
    @test Set(e.degree[1] for e in set) == Set(1:3)
    @test Set(e.degree[2] for e in set) == Set(2:3)
    # the restriction runs on both space bases
    @test Set(typeof(e.x_boundary) for e in set) == Set([Dirichlet, Free])
    @test length(Set(parameters(e.problem).c for e in set)) == length(set)
    @test count(e -> !iszero(timespan(e.problem)[1]), set) ≥ 2
    @test count(e -> rule_lengths(e)[1] > rule_lengths(e)[2], set) ≥ 2

    # the general problems: side data that vary in time, non-zero, and different per side and
    # per component
    general = [e for e in set if e.kind === :general]
    @test !isempty(general)
    for e in general
        t₀ = timespan(e.problem)[1]
        t₁ = t₀ + timestep(e.problem)
        for d in 1:components(e.problem)
            ga, gb = sides(e, t₀, d)
            ga₁, gb₁ = sides(e, t₁, d)
            @test abs(ga) ≥ floor && abs(gb) ≥ floor
            @test ga != gb
            @test ga != ga₁ && gb != gb₁
        end
        if components(e.problem) == 2
            @test sides(e, t₀, 1)[1] != sides(e, t₀, 2)[1]
            @test sides(e, t₀, 1)[2] != sides(e, t₀, 2)[2]
        end
    end

    # the Noether problems: rotation-invariant, zero side data, initial data zero at both ends,
    # and u² not proportional to u¹
    noether = [e for e in set if e.kind === :noether]
    @test length(noether) ≥ 4
    for e in noether
        @test components(e.problem) == 2
        @test e.problem.density isa VectorWave.VectorDensity
        t₀ = timespan(e.problem)[1]
        @test all(d -> sides(e, t₀, d) == (zero(T), zero(T)), 1:2)
        a, b = xspan(e.problem)
        for d in 1:2
            u, v, _ = component(e, [a, b], d)
            @test maximum(abs, u) ≤ 16 * eps(T)
        end
        x = xgrid(e.problem)
        u¹, _, _ = component(e, x, 1)
        u², _, _ = component(e, x, 2)
        cosine = abs(sum(u¹ .* u²)) / sqrt(sum(abs2, u¹) * sum(abs2, u²))
        @test cosine ≤ 1 - floor
    end

    # the symmetric problems: data symmetric under x ↦ a + b − x, not constant, not zero at the
    # ends, and equal side data
    symmetric = [e for e in set if e.kind === :symmetric]
    @test length(symmetric) ≥ 3
    for e in symmetric
        a, b = xspan(e.problem)
        x = xgrid(e.problem)
        mirror = a .+ b .- x
        t₀ = timespan(e.problem)[1]
        for d in 1:components(e.problem)
            u, v, w = component(e, x, d)
            ū, v̄, w̄ = component(e, mirror, d)
            scale = maximum(abs, u) + maximum(abs, w)
            @test maximum(abs, u - ū) ≤ 64 * eps(T) * scale
            @test maximum(abs, v - v̄) ≤ 64 * eps(T) * scale
            @test maximum(abs, w + w̄) ≤ 64 * eps(T) * scale
            @test maximum(u) - minimum(u) ≥ floor
            @test abs(first(u)) ≥ floor && abs(last(u)) ≥ floor
            ga, gb = sides(e, t₀, d)
            @test ga == gb && abs(ga) ≥ floor
        end
    end
end
