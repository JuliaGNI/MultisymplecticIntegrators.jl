using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using MultisymplecticIntegrators
using Random
using SimpleSplines
using Test

import GeometricEquations: problem
import GeometricIntegratorsBase: GeometricIntegrator, cache, solutionstep

const M = MultisymplecticIntegrators

include("../helpers/testsetup.jl")
include("../helpers/action_oracle.jl")

"""A coupling that implements none of the seam, so that each missing function is refused."""
struct Bare <: SlabCoupling end

"""A coupling that declares its free rows and nothing else, so that its residual is refused."""
struct RowsOnly <: SlabCoupling end

M.free_rows(::RowsOnly, a::SpaceTimeAnsatz) = M.ntime_functions(a)

"""
A coupling that restricts the bottom row as `CoefficientRestriction` does, and lifts the field by
the known function [`lift`](@ref) through the coupling's transform of the field.
"""
struct Lifted <: SlabCoupling end

M.free_rows(::Lifted, a::SpaceTimeAnsatz) = M.ntime_functions(a) - 1

function M.coupling_data(c::Lifted, a::TensorBSplineAnsatz, p::LagrangianPDEProblem,
        ::Type{ST}) where {ST}
    return M.RestrictionData{ST}(c, M.nspace_functions(a), components(p))
end

function M.coupling_data_type(::Lifted, ::TensorBSplineAnsatz, ::LagrangianPDEProblem,
        ::Type{ST}) where {ST}
    return M.RestrictionData{ST, Lifted}
end

"""
    lift(t̂, x, d)

The lift of component `d` at `(t̂, x)`, and its two derivatives `∂_{t̂}` and `∂ₓ`: a polynomial that
is not zero at the top of the slab, nor at an interior point, and that differs per component.
"""
function lift(t̂, x, d)
    a = d + t̂ + t̂^2
    b = x * (1 - x) / 4
    return (a * b, (1 + 2t̂) * b, a * (1 - 2x) / 4)
end

function M.transform_field!(u, v, w, ::M.RestrictionData{ST, Lifted}, t̂, x, h,
        problem) where {ST}
    for d in axes(u, 3), r in axes(u, 2), q in axes(u, 1)
        ℓ, ℓt, ℓx = lift(t̂[q], x[r], d)
        u[q, r, d] += ℓ
        v[q, r, d] += ℓt / h
        w[q, r, d] += ℓx
    end
    return nothing
end

# The field of the top row of the coefficient array and its two derivatives at the points `x`, by
# SimpleSplines' own evaluation, component-major, without the lift.
function top_field(c, a::TensorBSplineAnsatz{T}, x, h) where {T}
    tb = SimpleSplines.bases(SimpleSplines.basis(a))[1]
    xb = SimpleSplines.bases(SimpleSplines.basis(a))[2]
    coeffs = c.ansatz.coeffs
    nt, nx, D = size(coeffs)
    u = zeros(T, length(x), D)
    v = zeros(T, length(x), D)
    w = zeros(T, length(x), D)
    for d in 1:D, j in 1:nx, (i, p) in enumerate(x)
        slope = sum(coeffs[k, j, d] * SimpleSplines.evaluate(tb, k, one(T), 1)
        for k in 1:nt)
        u[i, d] += coeffs[nt, j, d] * SimpleSplines.evaluate(xb, j, p)
        v[i, d] += slope / h * SimpleSplines.evaluate(xb, j, p)
        w[i, d] += coeffs[nt, j, d] * SimpleSplines.evaluate(xb, j, p, 1)
    end
    return u, v, w
end

function wave_ansatz(::Type{T}; ncells = (2, 20), degree = 2, n = 3,
        x_boundary = Dirichlet()) where {T}
    return TensorBSplineAnsatz(degree = degree, ncells = ncells,
        xspan = (zero(T), one(T)), n = n, x_boundary = x_boundary)
end

function wave_problem(::Type{T}; timespan = (zero(T), T(0.25)), timestep = T(0.05),
        xstep = T(0.05)) where {T}
    return Wave.problem(T = T, timespan = timespan, timestep = timestep, xstep = xstep)
end

@testset "The coupling seam of the slab core" begin
    @testset "S1a the two hooks and the layout" for T in (Float64, Float32)
        problem = wave_problem(T)
        dirichlet = wave_ansatz(T)
        free = wave_ansatz(T; x_boundary = Free())
        nt = M.ntime_functions(dirichlet)
        nx = M.nspace_functions(dirichlet)
        nf = M.nspace_functions(free)
        # a coupling that fixes the bottom row leaves Nt - 1 free rows, and no unknowns of its own
        restricted = SpaceTimeGalerkin(dirichlet, CoefficientRestriction())
        @test total_free_coefficients(restricted, problem) == (nt - 1) * nx
        @test length(cache(GeometricIntegrator(problem, restricted)).x) == (nt - 1) * nx
        # the weak couplings leave all Nt rows free, and add their multipliers behind them
        weak = SpaceTimeGalerkin(dirichlet, WeakInitialCondition())
        @test total_free_coefficients(weak, problem) == nt * nx + nx
        @test length(cache(GeometricIntegrator(problem, weak)).x) == nt * nx + nx
        full = SpaceTimeGalerkin(free, WeakInitialAndBoundaryConditions())
        @test total_free_coefficients(full, problem) == nt * nf + (nf - 2) + 2 * nt
        @test length(cache(GeometricIntegrator(problem, full)).x) ==
              nt * nf + (nf - 2) + 2 * nt
        # the block's own sizes are that count: the bottom multipliers and those of the two ends
        block = cache(GeometricIntegrator(problem, full)).coupling
        @test length(block.initial_multiplier) + length(block.boundary_multiplier) ==
              (nf - 2) + 2 * nt
        # every block carries its coupling
        @test coupling(block) === coupling(full)
        @test coupling(cache(GeometricIntegrator(problem, restricted)).coupling) ===
              coupling(restricted)
        # the core asks the coupling for its residual term: the restriction's momentum load reaches
        # the row of the bottom test function, and setting the load aside takes it away again
        int = GeometricIntegrator(problem, restricted)
        c = cache(int)
        solstep = solutionstep(int, GeometricSolution(problem)[0])
        M.slab_bottom!(c, c.coupling, problem, dirichlet, solstep)
        x = copy(c.x)
        b1 = similar(x)
        b2 = similar(x)
        p = parameters(problem)
        M.residual!(b1, x, nothing, p, int)
        flux = copy(c.coupling.flux)
        fill!(c.coupling.flux, zero(T))
        M.residual!(b2, x, nothing, p, int)
        c.coupling.flux .= flux
        @test any(x -> !iszero(x), flux)
        for d in 1:components(problem), j in 1:nx

            row = M.free_index(1, j, d, nt - 1, nx)
            @test b1[row] == b2[row] + flux[j, d]
        end
        # the coupling's load reaches the row of the bottom test function, and no other
        @test count(k -> b1[k] != b2[k], eachindex(b1)) == nx * components(problem)
    end

    @testset "S1b a coupling without the seam is refused, function by function" for T in (
        Float64, Float32)
        problem = wave_problem(T)
        # no free rows: the integrator cannot be built
        err = try
            GeometricIntegrator(problem, SpaceTimeGalerkin(wave_ansatz(T), Bare()))
            nothing
        catch e
            e
        end
        @test err isa MethodError
        @test err.f === M.free_rows
        # free rows and no term of the residual: the residual is refused, not run as a no-op
        int = GeometricIntegrator(problem, SpaceTimeGalerkin(wave_ansatz(T), RowsOnly()))
        c = cache(int)
        @test c.coupling isa M.NoCouplingData{RowsOnly}
        @test coupling(c.coupling) === RowsOnly()
        b = similar(c.x)
        err = try
            M.residual!(b, c.x, nothing, parameters(problem), int)
            nothing
        catch e
            e
        end
        @test err isa MethodError
        @test err.f === M.coupling_term!
        # the per-coupling declarations: one free_rows method per coupling, no default term
        @test !hasmethod(M.free_rows, Tuple{SlabCoupling, SpaceTimeAnsatz})
        @test !hasmethod(M.coupling_term!,
            Tuple{Any, Any, M.CouplingData, Any, Any, Any})
    end

    @testset "S1c the coupling's transform of the field, $T" for T in (Float64, Float32)
        for problem in (wave_problem(T; timespan = (zero(T), T(0.1))),
            VectorWave.problem(T = T, timespan = (zero(T), T(0.1)), timestep = T(0.05),
            xstep = T(0.05)))
            a = wave_ansatz(T)
            method = SpaceTimeGalerkin(a, Lifted())
            D = components(problem)
            # the residual is the gradient of the action of the lifted field
            int = GeometricIntegrator(problem, method)
            c = cache(int)
            solstep = solutionstep(int, GeometricSolution(problem)[0])
            M.slab_bottom!(c, c.coupling, problem, a, solstep)
            oracle = ActionOracle(problem, method; restricted = true, lift = lift)
            unlifted = ActionOracle(problem, method; restricted = true)
            datum = first_datum(oracle)
            θ = 2 .* rand(Xoshiro(D), T, length(c.x)) .- 1
            b = similar(θ)
            M.residual!(b, θ, nothing, parameters(problem), int)
            ∇S = oracle_residual(oracle, datum, θ)
            scale = maximum(abs, ∇S)
            @test scale ≥ T(1 // 100)
            @test maximum(abs, b - ∇S) ≤ 100 * eps(T) * scale
            # the lift is visible: the residual of the unlifted field is another one
            @test maximum(abs, b - oracle_residual(unlifted, datum, θ)) >
                  T(1 // 100) * scale
            # the trace and the solution are the ansatz's field at the top of the slab plus the lift
            result = integrate(int)
            h = timestep(problem)
            nodes = space_quadrature_nodes(a)
            u, v, w = top_field(c, a, nodes, h)
            for d in 1:D, (r, x) in enumerate(nodes)

                ℓ, ℓt, ℓx = lift(one(T), x, d)
                @test c.trace_u[r, d] ≈ u[r, d] + ℓ atol = 100 * eps(T)
                @test c.trace_v[r, d] ≈ v[r, d] + ℓt / h atol = 100 * eps(T) / h
                @test c.trace_w[r, d] ≈ w[r, d] + ℓx atol = 100 * eps(T)
            end
            grid = xgrid(problem)
            n = length(grid)
            ug, vg, wg = top_field(c, a, grid, h)
            final = ntime(result.solution)
            for d in 1:D, (i, x) in enumerate(grid)

                k = (d - 1) * n + i
                ℓ, ℓt, ℓx = lift(one(T), x, d)
                @test result.solution.u[final][k] ≈ ug[i, d] + ℓ atol = 100 * eps(T)
                @test result.solution.v[final][k] ≈ vg[i, d] + ℓt / h atol = 100 * eps(T) /
                                                                             h
                @test result.solution.w[final][k] ≈ wg[i, d] + ℓx atol = 100 * eps(T)
            end
            assert_no_upcast(result.solution.u, T)
        end
    end

    @testset "S1d the cache holds an ansatz block and a coupling block, $T" for T in (Float64,
        Float32)
        problem = wave_problem(T)
        for method in (SpaceTimeGalerkin(wave_ansatz(T), CoefficientRestriction()),
            SpaceTimeGalerkin(wave_ansatz(T), WeakInitialCondition()),
            SpaceTimeGalerkin(wave_ansatz(T; x_boundary = Free()),
            WeakInitialAndBoundaryConditions()),
            SpaceTimeGalerkin(wave_ansatz(T); initial_guess = FitTo(PreviousSlab())))
            c = cache(GeometricIntegrator(problem, method))
            names = fieldnames(typeof(c))
            # the guess keeps no scratch in the cache
            for field in (:sample, :fit, :initial_sample, :taylor, :weighted, :load, :fibre)
                @test field ∉ names
            end
            # what only the B-spline ansatz fills is in its block, and the restriction's datum in
            # the restriction's block
            for field in (:coeffs, :residual, :top, :tabulation, :evaluation, :bottom, :flux)
                @test field ∉ names
            end
            @test c.ansatz isa M.TensorBSplineSlab{T, T}
            @test c.coupling isa M.CouplingData
            @test coupling(c.coupling) === coupling(method)
        end
    end

    @testset "S1e the trace and the solution share one evaluation, $T" for T in (Float64, Float32)
        problem = VectorWave.problem(
            T = T, timespan = (zero(T), T(0.1)), timestep = T(0.05),
            xstep = T(0.05))
        int = GeometricIntegrator(problem, SpaceTimeGalerkin(wave_ansatz(T)))
        c = cache(int)
        s = c.ansatz
        s.coeffs .= 2 .* rand(Xoshiro(3), T, size(s.coeffs)) .- 1
        # the output grid moved onto the nodes of the trace, so both are read at the same points
        c.ansatz = M.TensorBSplineSlab{T, T}(
            s.coeffs, s.residual, s.top, s.tabulation, s.trace,
            s.trace, s.top_slope)
        nq, D = size(c.trace_u)
        u, v, w = zeros(T, nq * D), zeros(T, nq * D), zeros(T, nq * D)
        M.assign!(u, v, w, c, problem)
        @test reshape(u, nq, D) == c.trace_u
        @test reshape(v, nq, D) == c.trace_v
        @test reshape(w, nq, D) == c.trace_w
        @test any(!iszero, c.trace_v)
    end
end
