using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using MultisymplecticIntegrators
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache, integrate!, reset!,
                                 solutionstep

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")

const M = MultisymplecticIntegrators

function slab_run(::Type{T}; degree = 2, ncells = (2, 20), n = 3, guess = Continuation(),
        problem = Wave.problem(T = T, timespan = (zero(T), T(0.25)), timestep = T(0.05),
            xstep = T(0.05))) where {T}
    ansatz = TensorBSplineAnsatz(degree = degree, ncells = ncells,
        xspan = (zero(T), one(T)), n = n)
    method = SpaceTimeGalerkin(ansatz, CoefficientRestriction(); initial_guess = guess)
    int = GeometricIntegrator(problem, method)
    return problem, method, int, integrate(int)
end

"""
A `D = 2` problem whose density is the vector density of `VectorWave`, built through the density
protocol of `LagrangianPDEProblem`: the initial, boundary and exact data of `VectorWave.problem`,
and the quartic coupling `β`.
"""
function vector_density_problem(::Type{T}; β = 0.5, timespan = (zero(T), T(0.25)),
        timestep = T(0.05), xstep = T(0.05)) where {T}
    params = VectorWave.default_parameters()
    template = VectorWave.problem(T = T, params = params, timespan = timespan,
        timestep = timestep, xstep = xstep)
    return LagrangianPDEProblem(VectorWave.vector_density(; params = params, β = β);
        D = 2,
        ics_function = template.ics_function,
        bcs_function = template.bcs_function,
        exact_u = template.exact_u,
        exact_v = template.exact_v,
        exact_w = template.exact_w,
        timespan = template.timespan,
        timestep = template.timestep,
        xspan = template.xspan,
        xstep = template.xstep,
        params = params)
end

"""
    slab_momentum(problem, method)

The discrete momentum of every step, `J_h = Σⱼ (q¹ⱼ p²ⱼ − q²ⱼ p¹ⱼ)` over the top coefficient row
`q` and the top row `p` of the slab's weak form, which is the momentum the coupling carries out;
and the Newton iteration count of the step beside it.

The steps are driven one at a time, because the momentum of a step exists only while its cache does.
"""
function slab_momentum(problem, method)
    T = datatype(problem)
    int = GeometricIntegrator(problem, method)
    c = cache(int)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    momenta = T[]
    iterations = Int[]
    for n in 1:ntime(solution)
        reset!(solstep, timesteps(solution)[n])
        integrate!(solstep, int)
        push!(momenta,
            sum(c.top[j, 1] * c.flux[j, 2] - c.top[j, 2] * c.flux[j, 1]
            for j in axes(c.top, 1)))
        push!(iterations, c.iterations[end])
    end
    return momenta, iterations
end

@testset "Coefficient restriction" begin
    @testset "A1c the full-restriction reference" for T in (Float64, Float32)
        _, _, _, result = slab_run(T)
        reference = WAVE_GALERKIN_FULL_RESTRICTION_BSPLINE.u
        for n in 1:5
            @test maximum(abs.(result.solution.u[n] .- reference[n, :])) ≤
                  T(100) * newton_tolerance(T, SOLVER_OPTIONS)
        end
    end

    @testset "A3a the coupling carries the top trace across a step" for T in (Float64, Float32)
        problem, method, int, result = slab_run(T)
        c = cache(int)
        @test length(c.iterations) == 5
        @test all(≥(1), c.iterations)
        @test length(c.residuals) == 5
        reference = WAVE_GALERKIN_FULL_RESTRICTION_BSPLINE.u
        for n in 1:5
            @test maximum(abs.(result.solution.u[n] .- reference[n, :])) ≤
                  T(100) * newton_tolerance(T, SOLVER_OPTIONS)
        end
        @test result.traces[end].u == vec(c.trace_u)
        @test length(result.traces) == 5
        # the top row of a one-step run is the bottom datum of the second step
        _, _, one_int, _ = slab_run(T;
            problem = Wave.problem(T = T,
                timespan = (zero(T), T(0.05)), timestep = T(0.05), xstep = T(0.05)))
        _, _, two_int, _ = slab_run(T;
            problem = Wave.problem(T = T,
                timespan = (zero(T), T(0.1)), timestep = T(0.05), xstep = T(0.05)))
        @test cache(two_int).bottom == cache(one_int).top
        # every entry of `residuals` is `‖F(x)‖₂` at its own step's final iterate, under the datum
        # and the momentum load that step started from, which the cache overwrites as the step ends
        stepped = GeometricIntegrator(problem, method)
        cs = cache(stepped)
        solution = GeometricSolution(problem)
        solstep = solutionstep(stepped, solution[0])
        p = parameters(problem)
        b = similar(cs.x)
        datum = similar(cs.bottom)
        load = similar(cs.flux)
        recomputed = Vector{T}(undef, ntime(solution))
        for n in 1:ntime(solution)
            # the datum and the momentum load the step will solve under
            M.slab_bottom!(cs, coupling(method), problem, ansatz(method), solstep)
            datum .= cs.bottom
            load .= cs.flux
            reset!(solstep, timesteps(solution)[n])
            integrate!(solstep, stepped)
            top_load = copy(cs.flux)
            cs.bottom .= datum
            cs.flux .= load
            M.residual!(b, cs.x, nothing, p, stepped)
            recomputed[n] = sqrt(sum(abs2, b))
            cs.flux .= top_load
        end
        for n in 1:5
            @test abs(recomputed[n] - cs.residuals[n]) ≤ T(100) * eps(T) * recomputed[n]
        end
        # a second run of one integrator starts again from the problem's initial data, and its
        # records belong to that run alone
        reused = GeometricIntegrator(problem, method)
        first_run = integrate(reused)
        second_run = integrate(reused)
        @test second_run.solution.u == first_run.solution.u
        @test length(cache(reused).iterations) == 5
        @test length(cache(reused).residuals) == 5
        @test length(cache(reused).initial_residuals) == 5
    end

    @testset "S3a the problem layer takes a vector density" for T in (Float64, Float32)
        problem = vector_density_problem(T; β = T(0.5))
        @test problem isa LagrangianPDEProblem
        @test components(problem) == 2
        @test problem.density isa VectorWave.VectorDensity
        # a density with a `params` field of its own gives the problem its parameters when the
        # keyword is absent; the `VectorDensity` above has none and takes them from the keyword
        template = VectorWave.problem(T = T, timespan = (zero(T), T(0.25)),
            timestep = T(0.05), xstep = T(0.05))
        density = Wave.density()
        inherited = LagrangianPDEProblem(density; D = 2,
            ics_function = (x, p) -> (u = zeros(T, 2 * length(x)),
                v = zeros(T, 2 * length(x)), w = zeros(T, 2 * length(x))),
            bcs_function = template.bcs_function,
            timespan = template.timespan, timestep = template.timestep,
            xspan = template.xspan, xstep = template.xstep)
        @test parameters(inherited) === density.params
    end

    @testset "S3b the vector density's derivatives carry the quartic term" for T in (Float64, Float32)
        β = T(0.5)
        params = VectorWave.default_parameters()
        scalar = VectorWave.density()
        vector = VectorWave.vector_density(; params = params, β = β)
        u = T[T(0.3), T(-0.7)]
        v = T[T(0.11), T(0.42)]
        w = T[T(-0.25), T(0.19)]
        dvec = M.derivatives(vector, u, v, w, params)
        r² = sum(abs2, u)
        @test r² > zero(T)
        for i in 1:2
            dsc = M.derivatives(scalar, u[i], v[i], w[i], params)
            @test abs(dvec.u[i] - (dsc.u - β * r² * u[i])) ≤ 16 * eps(T)
            @test abs(dvec.v[i] - dsc.v) ≤ 16 * eps(T)
            @test abs(dvec.w[i] - dsc.w) ≤ 16 * eps(T)
        end
    end

    @testset "A5a the two components and the discrete momentum" for T in (Float64, Float32)
        params = VectorWave.default_parameters()
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20),
            xspan = (zero(T), one(T)), n = 3)
        # m = β = 0: each component is its own scalar wave problem
        problem = VectorWave.problem(T = T, params = params, timespan = (zero(T), T(0.25)),
            timestep = T(0.05), xstep = T(0.05))
        result = integrate(problem, SpaceTimeGalerkin(ansatz, CoefficientRestriction()))
        @test components(problem) == 2
        assert_no_upcast(result.solution.u, T)
        n = length(xgrid(problem))
        for d in 1:2
            scalar = Wave.problem(
                T = T, params = VectorWave.component_parameters(params, d),
                timespan = (zero(T), T(0.25)), timestep = T(0.05), xstep = T(0.05))
            one_d = integrate(scalar, SpaceTimeGalerkin(ansatz, CoefficientRestriction()))
            for s in 1:5
                component = result.solution.u[s][((d - 1) * n + 1):(d * n)]
                @test maximum(abs.(component .- one_d.solution.u[s])) ≤
                      T(10) * eps(T) * maximum(abs, one_d.solution.u[s])
            end
        end
        # β ≠ 0: the quartic term couples the components, and the discrete momentum is conserved
        coupled = vector_density_problem(T; β = T(0.5), timespan = (zero(T), T(2.5)),
            timestep = T(0.05), xstep = T(0.05))
        momenta, iterations = slab_momentum(coupled,
            SpaceTimeGalerkin(ansatz, CoefficientRestriction()))
        @test length(momenta) == 50
        @test maximum(abs.(momenta .- momenta[1])) ≤ T(100) * T(8) * eps(T)
        @test maximum(iterations) > 1
    end
end
