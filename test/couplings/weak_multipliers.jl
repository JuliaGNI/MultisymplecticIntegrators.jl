using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: rank, svdvals
using MultisymplecticIntegrators
using SimpleSplines
using Test

import GeometricEquations: problem
import GeometricIntegratorsBase: GeometricIntegrator, cache, copy_internal_variables!,
                                 integrate!, internal, solutionstep

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")

const M = MultisymplecticIntegrators

"""
A coupling that implements none of the seam: it exists so that the core's fallback is reached, and
so that the refusal names it.
"""
struct NoopCoupling <: SlabCoupling end

"""
The bound of a reference comparison of this file, `800 eps(T)`, as the multiple of the Newton
tolerance the oracle records: `newton_tolerance(T, SOLVER_OPTIONS)` is `8 eps(T)`.
"""
reference_tolerance(::Type{T}) where {T} = T(100) * newton_tolerance(T, SOLVER_OPTIONS)

function wave_ansatz(::Type{T}; ncells = (2, 20), degree = 2, n = 3,
        x_boundary = Dirichlet()) where {T}
    return TensorBSplineAnsatz(degree = degree, ncells = ncells,
        xspan = (zero(T), one(T)), n = n, x_boundary = x_boundary)
end

function wave_problem(::Type{T}; timespan = (zero(T), T(0.25)), timestep = T(0.05),
        xstep = T(0.05)) where {T}
    return Wave.problem(T = T, timespan = timespan, timestep = timestep, xstep = xstep)
end

"""A whole run of one coupling on `Wave` at its default parameters: five steps of `h = 0.05`."""
function wave_run(::Type{T}; coupling = WeakInitialCondition(), x_boundary = Dirichlet(),
        ncells = (2, 20), guess = Continuation(),
        problem = wave_problem(T)) where {T}
    ansatz = wave_ansatz(T; ncells = ncells, x_boundary = x_boundary)
    method = SpaceTimeGalerkin(ansatz, coupling; initial_guess = guess)
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
    multiplier_momentum(problem, method)

The discrete momentum of every step, `J = Σᵣ ωᵣ (u¹ᵣ λ²ᵣ − u²ᵣ λ¹ᵣ)`, over the top trace `u` at the
nodes of the space rule, its weights `ω`, and the top multipliers `λ` there.

The steps are driven one at a time, because the multipliers of a step exist only while its cache
does.
"""
function multiplier_momentum(problem, method)
    T = datatype(problem)
    int = GeometricIntegrator(problem, method)
    c = cache(int)
    ω = c.tabulation.space.weight
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    momenta = T[]
    for n in 1:ntime(solution)
        reset!(solstep, timesteps(solution)[n])
        integrate!(solstep, int)
        value = zero(T)
        for r in axes(c.coupling.top_values, 1)
            value += ω[r] * (c.trace_u[r, 1] * c.coupling.top_values[r, 2] -
                      c.trace_u[r, 2] * c.coupling.top_values[r, 1])
        end
        push!(momenta, value)
    end
    return momenta
end

"""One finite difference per unknown of the slab's residual, at the state the cache holds."""
function slab_jacobian(int)
    x = copy(nlsolution(int))
    T = eltype(x)
    b₀ = similar(x)
    b₁ = similar(x)
    p = parameters(problem(int))
    M.residual!(b₀, x, nothing, p, int)
    J = zeros(T, length(x), length(x))
    δ = sqrt(eps(T))
    for k in eachindex(x)
        x[k] += δ
        M.residual!(b₁, x, nothing, p, int)
        x[k] -= δ
        for i in eachindex(x)
            J[i, k] = (b₁[i] - b₀[i]) / δ
        end
    end
    return J
end

"""The allocations of one residual evaluation, after one warm-up call."""
function residual_allocations(b, x, p, int)
    M.residual!(b, x, nothing, p, int)
    return @allocated M.residual!(b, x, nothing, p, int)
end

"""
A `Wave` problem whose exact solution is the travelling pulse `f(x − ct)` of non-zero value at both
ends of `xspan` over the run, so that the boundary data of every slab is inhomogeneous.
"""
function travelling_wave(::Type{T}; timespan = (zero(T), T(0.25)), timestep = T(0.05),
        xstep = T(0.05)) where {T}
    params = Wave.default_parameters()
    width = T(0.35)
    centre = T(0.5)
    profile(ξ) = exp(-(ξ - centre)^2 / (2 * width^2))
    slope(ξ) = -(ξ - centre) / width^2 * profile(ξ)
    speed(p) = convert(T, p.c)
    exact_u(t, x, p) = profile(x - speed(p) * t)
    exact_v(t, x, p) = -speed(p) * slope(x - speed(p) * t)
    exact_w(t, x, p) = slope(x - speed(p) * t)
    grid = collect(zero(T):xstep:one(T))
    ics = (x, p) -> field_initial_data(exact_u, exact_v, exact_w, x, p, 1, timespan[1])
    bcs = (times, p) -> field_boundary_data(exact_u, exact_v, exact_w, times, grid, p, 1)
    return LagrangianPDEProblem(Wave.density(; params = params); D = 1,
        ics_function = ics,
        bcs_function = bcs,
        exact_u = exact_u,
        exact_v = exact_v,
        exact_w = exact_w,
        timespan = timespan,
        timestep = timestep,
        xspan = (zero(T), one(T)),
        xstep = xstep,
        params = params)
end

"""
A `Wave` problem whose exact solution is the plane wave `u(t, x) = x − c t`, which the degree-2
basis represents exactly and whose boundary flux `−c²` is not zero: the data at the two ends of
`xspan` is `−c t` and `1 − c t`, so both boundary rows carry a value that varies over the slab.
"""
function plane_wave(::Type{T}; timespan = (zero(T), T(0.25)), timestep = T(0.05),
        xstep = T(0.05)) where {T}
    params = Wave.default_parameters()
    speed = convert(T, params.c)
    exact_u(t, x, p) = x - speed * t
    exact_v(t, x, p) = zero(x) + zero(t) - speed
    exact_w(t, x, p) = one(x) + zero(t)
    grid = collect(zero(T):xstep:one(T))
    ics = (x, p) -> field_initial_data(exact_u, exact_v, exact_w, x, p, 1, timespan[1])
    bcs = (times, p) -> field_boundary_data(exact_u, exact_v, exact_w, times, grid, p, 1)
    return LagrangianPDEProblem(Wave.density(; params = params); D = 1,
        ics_function = ics,
        bcs_function = bcs,
        exact_u = exact_u,
        exact_v = exact_v,
        exact_w = exact_w,
        timespan = timespan,
        timestep = timestep,
        xspan = (zero(T), one(T)),
        xstep = xstep,
        params = params)
end

@testset "Weak multipliers" begin
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
        full = SpaceTimeGalerkin(free, WeakInitialAndBoundaryConditions(stride = 1))
        @test total_free_coefficients(full, problem) == nt * nf + (nf - 2) + 2 * nt
        @test length(cache(GeometricIntegrator(problem, full)).x) ==
              nt * nf + (nf - 2) + 2 * nt
        # the unknowns of the block are the same count, reached from the cache
        @test M.coupling_unknowns(cache(GeometricIntegrator(problem, full)).coupling) ==
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
            # the coupling's load is added to the row, and to nothing else
            @test b1[row] == b2[row] + flux[j, d]
        end
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
    end

    @testset "A1a the Dirichlet reference" for T in (Float64, Float32)
        _, _, int, result = wave_run(T)
        reference = WAVE_GALERKIN_DIRICHLET_BSPLINE.u
        @test length(result.traces) == 5
        @test minimum(cache(int).iterations) ≥ 1
        for n in 1:5
            @test maximum(abs.(result.solution.u[n] .- reference[n, :])) ≤
                  reference_tolerance(T)
        end
        assert_no_upcast(result.solution.u, T)
    end

    @testset "A1b every other boundary condition is refused" for T in (Float64, Float32)
        problem = wave_problem(T)
        # each coupling accepts one condition — the one it needs — and refuses the other five,
        # naming the coupling, the condition it needs and the condition it got
        for (coupling, needed, accepted, refused) in (
            (WeakInitialCondition(), "Dirichlet", Dirichlet(),
            (Free(), Natural(), Neumann(), Periodic(), Robin(1, 1))),
            (WeakInitialAndBoundaryConditions(), "Free", Free(),
            (Dirichlet(), Natural(), Neumann(), Periodic(), Robin(1, 1))))
            @test GeometricIntegrator(problem,
                SpaceTimeGalerkin(wave_ansatz(T; x_boundary = accepted),
                    coupling)) isa GeometricIntegrator
            for b in refused
                err = try
                    GeometricIntegrator(problem,
                        SpaceTimeGalerkin(wave_ansatz(T; x_boundary = b), coupling))
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin(string(nameof(typeof(coupling))), err.msg)
                @test occursin(needed, err.msg)
                @test occursin(string(nameof(typeof(b))), err.msg)
            end
        end
    end

    @testset "A1c the momentum carry" for T in (Float64, Float32)
        problem, method, _, _ = wave_run(T)
        solution = GeometricSolution(problem)
        # one integrate_step! after another
        stepped = GeometricIntegrator(problem, method)
        cs = cache(stepped)
        solstep = solutionstep(stepped, solution[0])
        reset!(solstep, timesteps(solution)[1])
        integrate!(solstep, stepped)
        top = copy(cs.coupling.top_multiplier)
        @test any(x -> !iszero(x), top)
        reset!(solstep, timesteps(solution)[2])
        integrate!(solstep, stepped)
        @test cs.coupling.initial_multiplier == top
        # the same through the ecosystem's hook, on a cache that just finished a step
        carried = GeometricIntegrator(problem, method)
        cc = cache(carried)
        other = solutionstep(carried, solution[0])
        reset!(other, timesteps(solution)[1])
        integrate!(other, carried)
        copy_internal_variables!(other, cc)
        @test internal(other).top_multiplier == cc.coupling.top_multiplier
        M.slab_bottom!(cc, coupling(method), problem, ansatz(method), other)
        @test cc.coupling.initial_multiplier == cc.coupling.top_multiplier
        # through integrate: the datum of a run's last slab is the previous slab's top multiplier,
        # so a run of five slabs enters its fifth with what a run of four slabs handed out
        _, four_method, five_int, _ = wave_run(T)
        four_int = GeometricIntegrator(
            wave_problem(T; timespan = (zero(T), T(0.2))), four_method)
        integrate(four_int)
        @test cache(five_int).coupling.initial_multiplier ==
              cache(four_int).coupling.top_multiplier
        @test any(x -> !iszero(x), cache(four_int).coupling.top_multiplier)
    end

    @testset "A1d the rank of the first slab" for T in (Float64, Float32)
        # the rank tolerance is the one the coupled layout leaves measurable in each precision
        rank_tolerance = T === Float64 ? T(1e-10) : eps(T) / 16
        problem = wave_problem(T; timespan = (zero(T), T(0.05)))
        method = SpaceTimeGalerkin(wave_ansatz(T),
            WeakInitialCondition(multiplier_basis = :bspline))
        int = GeometricIntegrator(problem, method)
        integrate(int)
        unknown = length(cache(int).x)
        jacobian = slab_jacobian(int)
        σ = svdvals(jacobian)
        @info "A1d the rank of the Dirichlet slab" T unknown smallest=σ[end] relative=σ[end]/σ[1]
        @test rank(jacobian; rtol = rank_tolerance) == unknown
        # the same measurement for the coupling that imposes the boundary conditions; its λ basis
        # has lost the two corner functions, which is §Audit 2.8's layout
        full = SpaceTimeGalerkin(wave_ansatz(T; x_boundary = Free()),
            WeakInitialAndBoundaryConditions(stride = 1))
        full_int = GeometricIntegrator(problem, full)
        integrate(full_int)
        full_unknown = length(cache(full_int).x)
        full_jacobian = slab_jacobian(full_int)
        full_σ = svdvals(full_jacobian)
        @info "A1d the rank of the free slab" T unknown=full_unknown smallest=full_σ[end] relative=full_σ[end]/full_σ[1]
        @test full_unknown == 116
        @test rank(full_jacobian; rtol = rank_tolerance) == full_unknown
    end

    @testset "A1e a problem without an exact solution" for T in (Float64, Float32)
        template = wave_problem(T; timespan = (zero(T), T(0.1)))
        bare = LagrangianPDEProblem(template.density; D = 1,
            ics_function = template.ics_function,
            bcs_function = template.bcs_function,
            timespan = template.timespan, timestep = template.timestep,
            xspan = template.xspan, xstep = template.xstep,
            params = parameters(template))
        @test bare.exact_u === nothing
        method = SpaceTimeGalerkin(wave_ansatz(T), WeakInitialCondition())
        with_solution = integrate(template, method)
        without = integrate(bare, method)
        @test without.solution.u == with_solution.solution.u
        assert_no_upcast(without.solution.u, T)
    end

    @testset "A1f the nonlinear problem" for T in (Float64, Float32)
        errors = T[]
        for cells in (10, 20, 40)
            problem = SineGordon.problem(T = T, timespan = (zero(T), T(0.15)),
                timestep = T(0.05), xstep = T(0.05))
            ansatz = wave_ansatz(T; ncells = (2, cells), x_boundary = Free())
            method = SpaceTimeGalerkin(ansatz,
                WeakInitialAndBoundaryConditions(stride = 1, multiplier_basis = :bspline))
            int = GeometricIntegrator(problem, method)
            result = integrate(int)
            c = cache(int)
            @test length(result.traces) == 3
            @test length(c.iterations) == 3
            @test all(value -> isfinite(value), c.residuals)
            assert_no_upcast(result.solution.u, T)
            exact = [SineGordon.exact_u(problem.timespan[2], x, parameters(problem))
                     for x in xgrid(problem)]
            push!(errors, maximum(abs.(result.solution.u[end] .- exact)))
        end
        orders = [log2(errors[i] / errors[i + 1]) for i in 1:2]
        @info "A1f the observed order" errors=errors orders=orders
        @test all(≥(T(1.9)), orders)
    end

    @testset "A2a the plane wave" for T in (Float64, Float32)
        problem = plane_wave(T)
        method = SpaceTimeGalerkin(wave_ansatz(T; x_boundary = Free()),
            WeakInitialAndBoundaryConditions(stride = 1))
        int = GeometricIntegrator(problem, method)
        result = integrate(int)
        p = parameters(problem)
        @test length(result.traces) == 5
        for n in 1:5
            exact = [problem.exact_u(timesteps(result.solution)[n], x, p)
                     for x in xgrid(problem)]
            @test maximum(abs.(result.solution.u[n] .- exact)) ≤ reference_tolerance(T)
        end
        assert_no_upcast(result.solution.u, T)
    end

    @testset "A2b the stride and the multiplier counts" for T in (Float64, Float32)
        problem = wave_problem(T)
        ansatz = wave_ansatz(T; x_boundary = Free())
        nx = M.nspace_functions(ansatz)
        nt = M.ntime_functions(ansatz)
        for (stride, nλ, nμ) in ((1, nx - 2, nt), (2, cld(nx, 2), cld(nt, 2)))
            for basis in (:bspline, :lagrange)
                coupling = WeakInitialAndBoundaryConditions(stride = stride,
                    multiplier_basis = basis)
                c = cache(GeometricIntegrator(problem, SpaceTimeGalerkin(ansatz, coupling)))
                @test size(c.coupling.initial_multiplier, 1) == nλ
                @test size(c.coupling.boundary_basis, 1) == nμ
                @test M.coupling_unknowns(c.coupling) == (nλ + 2 * nμ) * components(problem)
            end
        end
        # the two functions that leave the bottom surface's basis at stride = 1 are the corners
        for basis in (:bspline, :lagrange)
            first_ = WeakInitialAndBoundaryConditions(stride = 1, multiplier_basis = basis)
            @test setdiff(collect(1:nx), M.space_multiplier_indices(first_, ansatz)) ==
                  [1, nx]
            second = WeakInitialAndBoundaryConditions(stride = 2, multiplier_basis = basis)
            @test M.space_multiplier_indices(second, ansatz) == collect(1:2:nx)
            @test M.boundary_multiplier_indices(first_, ansatz) == collect(1:nt)
            @test M.boundary_multiplier_indices(second, ansatz) == collect(1:2:nt)
        end
        @test_throws ArgumentError WeakInitialAndBoundaryConditions(stride = 0)
        @test_throws ArgumentError WeakInitialCondition(multiplier_basis = :BSplineDirichlet)
        @test_throws ArgumentError WeakInitialAndBoundaryConditions(multiplier_basis = :Lagrange)
    end

    @testset "A3a the two bases agree as the mesh refines" for T in (Float64, Float32)
        errors = Dict{Symbol, Vector{T}}()
        solutions = Dict{Symbol, Vector{T}}()
        tables = Dict{Symbol, Matrix{T}}()
        for basis in (:bspline, :lagrange)
            errors[basis] = T[]
            for cells in (5, 10, 20)
                problem = wave_problem(T; timestep = T(0.25 / (2 * cells / 5)),
                    xstep = one(T) / cells)
                coupling = WeakInitialAndBoundaryConditions(stride = 1,
                    multiplier_basis = basis)
                int = GeometricIntegrator(problem,
                    SpaceTimeGalerkin(
                        wave_ansatz(T; ncells = (2, cells), x_boundary = Free()),
                        coupling))
                result = integrate(int)
                exact = [Wave.exact_u(problem.timespan[2], x, parameters(problem))
                         for x in xgrid(problem)]
                push!(errors[basis], maximum(abs.(result.solution.u[end] .- exact)))
                # the run's own solution and the table it was assembled in; the finest one stays
                solutions[basis] = copy(result.solution.u[end])
                tables[basis] = copy(cache(int).coupling.multiplier_basis)
                assert_no_upcast(result.solution.u, T)
            end
        end
        orders = map(basis -> [log2(errors[basis][i] / errors[basis][i + 1]) for i in 1:2],
            (:bspline, :lagrange))
        @info "A3a the observed orders" T errors=errors orders=orders
        @test all(≥(T(1.9)), orders[1])
        @test all(≥(T(1.9)), orders[2])
        # the two bases span one space in different functions: the tables differ, and the two runs
        # are the solution of that one space to round-off
        @test size(tables[:bspline]) == size(tables[:lagrange])
        @test tables[:bspline] != tables[:lagrange]
        @test maximum(abs.(solutions[:bspline] .- solutions[:lagrange])) ≤
              reference_tolerance(T)
    end

    @testset "A4a the guess lives in the guess" for T in (Float64, Float32)
        problem, _, int, _ = wave_run(T; coupling = WeakInitialAndBoundaryConditions(),
            x_boundary = Free())
        c = cache(int)
        b₁ = similar(c.x)
        b₂ = similar(c.x)
        p = parameters(problem)
        M.residual!(b₁, c.x, nothing, p, int)
        M.residual!(b₂, c.x, nothing, p, int)
        @test b₁ == b₂
    end

    @testset "A5a the residual allocates nothing" for T in (Float64, Float32)
        problem, _, int, _ = wave_run(T; coupling = WeakInitialAndBoundaryConditions(),
            x_boundary = Free())
        c = cache(int)
        b = similar(c.x)
        p = parameters(problem)
        @test residual_allocations(b, c.x, p, int) == 0
        @test hasproperty(c.coupling, :projection)
        @test size(c.coupling.projection, 1) == size(c.coupling.initial_multiplier, 1)
    end

    @testset "A6a a travelling wave with non-zero boundary values" for T in (Float64, Float32)
        errors = T[]
        for cells in (10, 20, 40)
            problem = travelling_wave(T; timestep = T(0.25 / (2 * cells / 5)),
                xstep = one(T) / cells)
            result = integrate(problem,
                SpaceTimeGalerkin(wave_ansatz(T; ncells = (2, cells), x_boundary = Free()),
                    WeakInitialAndBoundaryConditions(stride = 1)))
            exact = [problem.exact_u(problem.timespan[2], x, parameters(problem))
                     for x in xgrid(problem)]
            push!(errors, maximum(abs.(result.solution.u[end] .- exact)))
            assert_no_upcast(result.solution.u, T)
        end
        orders = [log2(errors[i] / errors[i + 1]) for i in 1:2]
        @info "A6a the observed order" T errors=errors orders=orders
        @test all(≥(T(1.9)), orders)
        @test errors[end] < T(1e-4)
    end

    @testset "A7a the two components and the discrete momentum" for T in (Float64, Float32)
        problem = vector_density_problem(T; β = T(0.5), timespan = (zero(T), T(2.5)))
        ansatz = wave_ansatz(T; ncells = (1, 20), x_boundary = Free())
        method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions(stride = 1))
        momenta = multiplier_momentum(problem, method)
        @test length(momenta) == 50
        @test maximum(abs.(momenta .- momenta[1])) ≤ reference_tolerance(T)
        int = GeometricIntegrator(problem, method)
        result = integrate(int)
        assert_no_upcast(result.solution.u, T)
        c = cache(int)
        @test c.coupling.top_multiplier[:, 1] != c.coupling.top_multiplier[:, 2]
        @test any(x -> !iszero(x), c.coupling.top_values[:, 1])
    end

    @testset "A8a the two precisions and the record" begin
        @test :WeakInitialCondition in names(MultisymplecticIntegrators)
        @test :WeakInitialAndBoundaryConditions in names(MultisymplecticIntegrators)
        # the signature is the docstring's first line, before the prose
        for name in (:WeakInitialCondition, :WeakInitialAndBoundaryConditions)
            binding = Base.Docs.Binding(MultisymplecticIntegrators, name)
            documented = join(Base.Docs.docstr(binding).text)
            first_line = strip(first(filter(!isempty, split(documented, '\n'))))
            @test startswith(first_line, string(name))
            @test occursin("<: SlabCoupling", first_line)
        end
        # the state a step carries between slabs is of the problem's element type as well
        for T in (Float64, Float32)
            problem, _, int, _ = wave_run(T)
            solstep = solutionstep(int, GeometricSolution(problem)[0])
            copy_internal_variables!(solstep, cache(int))
            assert_no_upcast(internal(solstep).top_multiplier, T)
        end
    end

    @testset "the coupling's buffers hold the longer rule" for T in (Float64, Float32)
        # two time cells against one space cell: the time rule has more nodes than the space rule,
        # and the trace a step guesses at the two ends of xspan is taken on the time nodes
        problem = wave_problem(T; timespan = (zero(T), T(0.1)))
        int = GeometricIntegrator(problem,
            SpaceTimeGalerkin(wave_ansatz(T; ncells = (2, 1), x_boundary = Free()),
                WeakInitialAndBoundaryConditions(stride = 1)))
        c = cache(int)
        @test size(c.coupling.top_u, 1) ==
              max(length(c.tabulation.time.node), length(c.tabulation.space.node))
        result = integrate(int)
        @test length(result.traces) == 2
        @test all(>(0), c.iterations)
        @test any(x -> !iszero(x), c.coupling.top_values)
    end
end
