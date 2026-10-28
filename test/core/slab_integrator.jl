using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using MultisymplecticIntegrators
using SimpleSplines
using Test

import GeometricEquations: problem
import GeometricIntegratorsBase: GeometricIntegrator, cache, solutionstep

const M = MultisymplecticIntegrators

"""
A coupling that implements none of the seam: it exists so that the core's fallback is reached, and
so that the refusal names it.
"""
struct NoopCoupling <: SlabCoupling end

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
        # the core asks the coupling for its residual term: the restriction's momentum load reaches
        # the row of the bottom test function, and setting the load aside takes it away again
        int = GeometricIntegrator(problem, restricted)
        c = cache(int)
        solstep = solutionstep(int, GeometricSolution(problem)[0])
        M.slab_bottom!(c, coupling(restricted), problem, dirichlet, solstep)
        x = copy(c.x)
        b1 = similar(x)
        b2 = similar(x)
        p = parameters(problem)
        M.residual!(b1, x, nothing, p, int)
        flux = copy(c.flux)
        fill!(c.flux, zero(T))
        M.residual!(b2, x, nothing, p, int)
        c.flux .= flux
        @test any(x -> !iszero(x), flux)
        for d in 1:components(problem), j in 1:nx

            row = M.free_index(1, j, d, nt - 1, nx)
            @test b1[row] == b2[row] + flux[j, d]
        end
        # the coupling's load reaches the row of the bottom test function, and no other
        @test count(k -> b1[k] != b2[k], eachindex(b1)) == nx * components(problem)
    end

    @testset "S1b a coupling that implements neither hook is refused" for T in (Float64, Float32)
        problem = wave_problem(T)
        coupling = NoopCoupling()
        method = SpaceTimeGalerkin(wave_ansatz(T), coupling)
        int = GeometricIntegrator(problem, method)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        err = try
            M.slab_bottom!(cache(int), coupling, problem, ansatz(method), solstep)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(string(nameof(typeof(coupling))), err.msg)
        # the same for the coupling's term of the residual: a multiplier coupling whose own method
        # the block does not match is refused rather than run as a no-op
        weak = WeakInitialCondition()
        cw = cache(GeometricIntegrator(problem, SpaceTimeGalerkin(wave_ansatz(T), weak)))
        b = similar(cw.x)
        err = try
            M.coupling_term!(b, cw.x, weak, M.NoCouplingData(), cw, wave_ansatz(T), problem)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(string(nameof(typeof(weak))), err.msg)
    end
end
