# The fixed-boundary coupling: the runs, the continuation, the predictor, the bubble fit, the element
# type, the two charts and a failed step. One of the files of `test/couplings/fixed_boundary*.jl`,
# which share `test/helpers/fixed_boundary_setup.jl`.
using GeometricBase: parameters, timestep, timespan
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: BLAS, ColumnNorm, diag, norm, qr
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache, current, integrate!, reset!,
                                 solutionstep, solver
import GeometricBase: ntime, timesteps

include("../helpers/problem_generator.jl")
include("../helpers/action_oracle.jl")
include("../helpers/testsetup.jl")
# The registry of the scripts and no part of the package: the family's rows are built here.
include(joinpath(@__DIR__, "..", "..", "scripts", "experiments.jl"))
include("../helpers/fixed_boundary_setup.jl")

# ---- FB9 -----------------------------------------------------------------------------

@testset "FB9 the dead branch on Wave over twenty steps" begin
    problem = WF.problem(timespan = (0.0, 1.0), timestep = 0.05, xstep = 0.05,
        xspan = (0.0, 1.0))
    ansatz = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = :abg, xspan = (0.0, 1.0),
        ncells = 20, nq = (6, 4))
    method = SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = Continuation())
    result = integrate(problem, method; solver = Newton(), max_iterations = 60,
        f_abstol = 0.0, f_reltol = fixed_boundary_f_reltol(Float64), verbosity = 0)
    stats = result.stats
    @test length(stats.converged) == 20
    @test all(stats.converged)
    t = result.solution.t
    u = result.solution.u
    grid = xspan(problem)
    xs = collect(range(grid[1], grid[2]; length = 21))
    error = maximum(abs(WF.solution_u(t[end], x, parameters(problem)) - u[end][i])
    for (i, x) in enumerate(xs))
    @info("FB9 the dead branch's twenty steps", VERSION, T="Float64", error=error,
        worst_ratio=maximum(stats.residuals ./ stats.initial_residuals),
        machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(),
        check_bounds=Base.JLOptions().check_bounds)
    @test 5.96e-4 ≤ error ≤ 7.28e-4
    # N20's `Float32` arm: two steps, recorded, and only the element types asserted
    problem32 = WF.problem(T = Float32, timespan = (0.0f0, 0.1f0), timestep = 0.05f0,
        xstep = 0.05f0, xspan = (0.0f0, 1.0f0))
    ansatz32 = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = :abg,
        xspan = (0.0f0, 1.0f0), ncells = 20, nq = (6, 4))
    method32 = SpaceTimeGalerkin(ansatz32, FixedBoundary(); initial_guess = Continuation())
    result32 = integrate(problem32, method32; solver = Newton(), max_iterations = 60,
        f_abstol = 0.0f0, f_reltol = fixed_boundary_f_reltol(Float32), verbosity = 0)
    @info("FB9 the dead branch's Float32 arm", VERSION, T="Float32",
        ratios=result32.stats.residuals ./ result32.stats.initial_residuals,
        machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(),
        check_bounds=Base.JLOptions().check_bounds)
    @test length(result32.stats.converged) == 2
    assert_no_upcast(result32.solution.u, Float32)
    @test all(isreal, result32.stats.residuals)
end

# ---- FB10 ----------------------------------------------------------------------------

@testset "FB10 the continuation starts at φ₀ and the previous Θ" begin
    for form in (:abg, :tsr)
        problem = PROBLEMS[:wave]()
        ansatz = fixed_ansatz(problem; form = form)
        slab = fixed_slab(problem, ansatz)
        c = cache(slab.int)
        nfp = length(M.spatial_free(ansatz))
        M.initial_guess!(current(slab.solstep), nothing, parameters(problem), slab.int)
        start = M.bubble_start(ansatz, Float64)
        @test copy(c.x[1:nfp]) ≈ c.coupling.phi[M.spatial_free(ansatz)]
        @test copy(c.x[(nfp + 1):end]) ≈ start[M.bubble_free(ansatz)]
        @test all(iszero, view(start, (3ansatz.St + 1):length(start)))
        integrate!(slab.solstep, slab.int)
        solved = copy(c.x)
        next_fixed_slab!(slab, 2)
        M.initial_guess!(current(slab.solstep), nothing, parameters(problem), slab.int)
        @test copy(c.x[1:nfp]) ≈ c.coupling.phi[M.spatial_free(ansatz)]
        @test copy(c.x[(nfp + 1):end]) ≈ solved[(nfp + 1):end]
    end
end

# ---- FB11 ----------------------------------------------------------------------------

@testset "FB11 a unit at r = 0" begin
    problem = PROBLEMS[:wave]()
    a = FixedBoundaryNetwork(Sx = 8, St = 9, bubble_form = :tsr, xspan = (0.0, 1.0),
        ncells = 3, nq = (4, 3))
    @test_throws r"^ArgumentError: the bubble's unit 5 has α = β = 0, that is r = 0, and no \(θ, s, ρ\) coordinates: exp\(ρ\) is never zero$" M.bubble_start(
        a, Float64)
    abg = FixedBoundaryNetwork(Sx = 8, St = 9, bubble_form = :abg, xspan = (0.0, 1.0),
        ncells = 3, nq = (4, 3))
    @test length(M.bubble_start(abg, Float64)) == M.nbubble(abg)
    method = SpaceTimeGalerkin(abg, FixedBoundary())
    result = integrate(problem, method; max_iterations = 2, verbosity = 0)
    @test isfinite(norm(result.solution.u[end]))
    atoms = bubble_atoms(BubbleGreedy(), Float64)
    @test size(atoms, 1) == 2184
    @test !any(iszero(atoms[i, 1]) && iszero(atoms[i, 2]) for i in axes(atoms, 1))
end

# ---- FB12 ----------------------------------------------------------------------------

@testset "FB12 the predictor against the test's own copy" begin
    for name in (:wave, :sinegordon)
        problem = PROBLEMS[name]()
        ansatz = fixed_ansatz(problem)
        slab = fixed_slab(problem, ansatz; max_iterations = 3, verbosity = 0)
        c = cache(slab.int)
        tb = ansatz.tabulation
        p = parameters(problem)
        h = timestep(problem)
        grid = collect(range(xspan(problem)[1], xspan(problem)[2]; length = 101))
        dx = grid[2] - grid[1]
        dt = h / 10
        # The density's one constant `m`, computed here rather than read from the code under test:
        # `∂L/∂v = m v`, so `m` is that derivative at `v = 1` with `u` and `w` at zero.
        mass = derivative_v(problem.density, 0.0, 1.0, 0.0, p)
        # The test's own Störmer–Verlet: ten substeps, central differences, the side data imposed at
        # every substep of the slab that starts at `t₀`, and nothing of the package's predictor but
        # the density's own derivatives.
        function stormer_verlet(u, v, t₀)
            accel(u, v) = begin
                a = zeros(length(u))
                w = similar(u)
                for j in eachindex(u)
                    j1, j2 = max(j - 1, 1), min(j + 1, length(u))
                    w[j] = (u[j2] - u[j1]) / ((j2 - j1) * dx)
                end
                du = zeros(length(u))
                dw = zeros(length(u))
                for j in eachindex(u)
                    du[j] = derivative_u(problem.density, u[j], v[j], w[j], p)
                    dw[j] = derivative_w(problem.density, u[j], v[j], w[j], p)
                end
                for j in 2:(length(u) - 1)
                    a[j] = (du[j] - (dw[j + 1] - dw[j - 1]) / (2dx)) / mass
                end
                a
            end
            U = zeros(11, length(u))
            U[1, :] .= u
            a = accel(u, v)
            for k in 1:10
                v .+= dt / 2 .* a
                u .+= dt .* v
                u[1] = problem.bcs_function(t₀ + k * dt, p).left.u[1]
                u[end] = problem.bcs_function(t₀ + k * dt, p).right.u[1]
                a = accel(u, v)
                v .+= dt / 2 .* a
                U[k + 1, :] .= u
            end
            return U
        end
        at(values, x) = begin
            j = clamp(floor(Int, (x - grid[1]) / dx) + 1, 1, length(grid) - 1)
            λ = (x - grid[j]) / dx
            (1 - λ) * values[j] + λ * values[j + 1]
        end
        sample = zeros(Float64, length(tb.tnode), length(tb.xnode), ansatz.D)
        function compare(U)
            for q in eachindex(tb.tnode), r in eachindex(tb.xnode)

                position = tb.tnode[q] * 10
                k = clamp(floor(Int, position) + 1, 1, 10)
                λ = position - (k - 1)
                expect = (1 - λ) * at(U[k, :], tb.xnode[r]) +
                         λ * at(U[k + 1, :], tb.xnode[r])
                @test abs(sample[q, r, 1] - expect) ≤
                      1e3 * eps(Float64) * max(1, abs(expect))
            end
            return nothing
        end
        # the first slab: from the datum's level and the initial data's velocity
        M.evaluate!(sample, Predictor(), c, ansatz, problem, current(slab.solstep))
        initial = problem.ics_function(grid, p)
        u = [level_network(problem, ansatz, c.coupling.phi, x, 0.0, 1) for x in grid]
        compare(stormer_verlet(u, collect(initial.v), 0.0))
        # the second slab: from the level the step produced and the velocity at that slab's top
        datum = copy(c.coupling.phi)
        integrate!(slab.solstep, slab.int)
        ns = M.nspatial(ansatz)
        φ1 = copy(c.ansatz.params[1:ns])
        Θ = copy(c.ansatz.params[(ns + 1):end])
        oracle = ActionOracle(problem, slab.method; block = c.ansatz)
        velocity = [top_velocity(oracle, problem, ansatz, datum, φ1, Θ, 0.0, x, 1)
                    for x in grid]
        next_fixed_slab!(slab, 2)
        @test c.coupling.phi == φ1
        M.evaluate!(sample, Predictor(), c, ansatz, problem, current(slab.solstep))
        compare(stormer_verlet([level_network(problem, ansatz, φ1, x, h, 1) for x in grid],
            velocity, h))
        @test isfinite(maximum(abs, c.ansatz.top_sample))
    end
    transport = M.LinearTransport.problem(timespan = (0.0, 0.1), timestep = 0.05,
        xstep = 0.05, xspan = (0.0, 1.0))
    ansatz = fixed_ansatz(transport; Sx = 2, St = 2)
    @test_throws r"^ArgumentError: Predictor\(\) needs a density whose ∂L/∂v" GeometricIntegrator(
        transport,
        SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = FitTo(Predictor())))
    # A density whose `∂L/∂v` is nonlinear vanishes at zero velocity and gives one value at
    # `v = 1` too: only a second velocity shows that its ratio is not the constant `m`.
    u_, v_, w_ = Symbolics.variable.((:u, :v, :w))
    cubic = LagrangianDensity(v_^3 / 3, u_, v_, w_)
    @test_throws r"^ArgumentError: Predictor\(\) needs one constant ∂L/∂v / v" M.predictor_mass(
        cubic, NamedTuple(), Float64, 1)
    wave = PROBLEMS[:wave]()
    spline = TensorBSplineAnsatz(degree = 3, ncells = (1, 5), xspan = (0.0, 1.0))
    @test_throws r"^ArgumentError: Predictor\(\) needs a FixedBoundaryNetwork; TensorBSplineAnsatz has no slab predictor$" GeometricIntegrator(
        wave, SpaceTimeGalerkin(spline, WeakInitialCondition();
            initial_guess = FitTo(Predictor())))
end

# ---- FB13 ----------------------------------------------------------------------------

@testset "FB13 a second bubble fit on one block reads its own dictionary" begin
    problem = PROBLEMS[:wave]()
    ansatz = fixed_ansatz(problem)
    slab = fixed_slab(problem, ansatz)
    c = cache(slab.int)
    tb = ansatz.tabulation
    sample = zeros(Float64, length(tb.tnode), length(tb.xnode), ansatz.D)
    M.evaluate!(sample, InitialCondition(), c, ansatz, problem, current(slab.solstep))
    w = collect(c.ansatz.params)
    M.fit!(w, BubbleGreedy(γ_range = (-6.0, 6.0)), ansatz, sample, c)
    M.fit!(w, BubbleGreedy(γ_range = (-3.0, 3.0)), ansatz, sample, c)
    second = copy(w)
    chosen = c.ansatz.fit.selected[1:ansatz.St]
    @test all(c.ansatz.fit.atoms[i, 3] ≤ 3.0 for i in chosen)
    @test c.ansatz.fit.method.γ_range == (-3.0, 3.0)
    c.ansatz.fit.method = nothing
    w2 = collect(c.ansatz.params)
    M.fit!(w2, BubbleGreedy(γ_range = (-3.0, 3.0)), ansatz, sample, c)
    @test second ≈ w2
end

# ---- FB14 ----------------------------------------------------------------------------

@testset "FB14 no silent upcast" begin
    problem = WF.problem(T = Float32, timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05,
        xspan = (0.0, 1.0))
    for form in (:abg, :tsr), solve in (:newton, :lm)

        a = fixed_ansatz(problem; form = form)
        m = SpaceTimeGalerkin(a, FixedBoundary())
        solvermethod = solve === :lm ? LevenbergMarquardt() : Newton()
        # The cache is read after two steps, not as `Cache` built it: the datum, the momentum the
        # steps handed on and the fit's dictionary are written by the run.
        int = GeometricIntegrator(problem, m; solver = solvermethod, max_iterations = 2,
            verbosity = 0)
        c = cache(int)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        for n in 1:2
            reset!(solstep, timesteps(solution)[n])
            M.slab_bottom!(c, c.coupling, problem, a, current(solstep))
            M.initial_guess!(current(solstep), nothing, parameters(problem), int)
            integrate!(solstep, int)
        end
        result = integrate(problem, m; solver = solvermethod, max_iterations = 2,
            verbosity = 0)
        assert_no_upcast(result.solution.u, Float32)
        assert_no_upcast(c.x, Float32)
        assert_no_upcast(c.coupling.momentum, Float32)
        assert_no_upcast(c.coupling.phi, Float32)
        assert_no_upcast(c.coupling.p_next, Float32)
        assert_no_upcast(c.coupling.phi_next, Float32)
        assert_no_upcast(c.coupling.v_top, Float32)
        assert_no_upcast(c.ansatz.params, Float32)
        assert_no_upcast(c.ansatz.momentum, Float32)
        assert_no_upcast(c.ansatz.top_sample, Float32)
        assert_no_upcast(c.ansatz.fit.atoms, Float32)
        assert_no_upcast(c.ansatz.fit.table, Float32)
        assert_no_upcast(c.u, Float32)
        assert_no_upcast(c.v, Float32)
        assert_no_upcast(c.w, Float32)
    end
end

# ---- FB15 ----------------------------------------------------------------------------

@testset "FB15 the two forms are one scheme" begin
    # N8: at the same greedy atoms the bubble is one bubble in two charts, so the field and its two
    # derivatives agree at every node, the `:tsr` residual is the `:abg` residual after the chart's
    # transpose, and no parameter of either form is wrapped into a range.
    problem = clause_problem(:wave)
    abg = clause_ansatz(problem; form = :abg)
    tsr = clause_ansatz(problem; form = :tsr)
    slab_abg = fixed_slab(problem, abg)
    slab_tsr = fixed_slab(problem, tsr)
    ca, ct = cache(slab_abg.int), cache(slab_tsr.int)
    p = parameters(problem)
    M.initial_guess!(current(slab_abg.solstep), nothing, p, slab_abg.int)
    M.initial_guess!(current(slab_tsr.solstep), nothing, p, slab_tsr.int)
    # the same greedy atoms in both blocks, each in its own coordinates: the dictionary is one
    # dictionary, so atom `k` is the same bubble in either form, and the output weights are one
    # nonzero number in both
    atoms = bubble_atoms(BubbleGreedy(), Float64)
    St, nsp = tsr.St, M.nspatial(tsr)
    for k in 1:St
        i = 1 + (k - 1) * 218
        α, β, γ = atoms[i, 1], atoms[i, 2], atoms[i, 3]
        ca.x[nsp + k], ca.x[nsp + St + k], ca.x[nsp + 2St + k] = α, β, γ
        θ, s, ρ = M.bubble_chart(α, β, γ, k)
        ct.x[nsp + k], ct.x[nsp + St + k], ct.x[nsp + 2St + k] = θ, s, ρ
    end
    first_weight = nsp + 3St + 1
    ca.x[first_weight:end] .= 0.3
    ct.x[first_weight:end] .= 0.3
    hidden = (nsp + 1):(nsp + 3St)
    b_abg = zeros(Float64, length(ca.x))
    b_tsr = zeros(Float64, length(ct.x))
    M.residual!(b_abg, ca.x, nothing, p, slab_abg.int)
    M.residual!(b_tsr, ct.x, nothing, p, slab_tsr.int)
    scale = max(1, maximum(abs, ct.u))
    @test maximum(abs, ca.u - ct.u) ≤ 1e3 * eps(Float64) * scale
    @test maximum(abs, ca.v - ct.v) ≤ 1e3 * eps(Float64) * maximum(abs, ct.v)
    @test maximum(abs, ca.w - ct.w) ≤ 1e3 * eps(Float64) * maximum(abs, ct.w)
    # the chart, written here from its definition: `(α, β, γ) = (e^ρ cos θ, e^ρ sin θ, −e^ρ s)`
    chart(θ, s, ρ) = (exp(ρ) * cos(θ), exp(ρ) * sin(θ), -exp(ρ) * s)
    nf = length(ct.x)
    G = zeros(Float64, nf, nf)
    for k in 1:nf
        G[k, k] = 1
    end
    for k in 1:St
        θ, s, ρ = ct.x[nsp + k], ct.x[nsp + St + k], ct.x[nsp + 2St + k]
        J = ForwardDiff.jacobian(z -> collect(chart(z[1], z[2], z[3])), [θ, s, ρ])
        for i in 1:3, j in 1:3

            G[nsp + (i - 1) * St + k, nsp + (j - 1) * St + k] = J[i, j]
        end
    end
    r_abg = slab_residual(slab_abg.int, ca.x, p)
    r_tsr = slab_residual(slab_tsr.int, ct.x, p)
    # the comparison below is evidence only while the field depends on the atoms and the residual's
    # hidden rows are not zero: the first step's guess writes the output weights as zero, and at zero
    # the chart's transpose blocks multiply nothing
    @test maximum(abs, ca.x[first_weight:end]) > 0
    @test maximum(abs, r_abg[hidden]) > 0
    @test maximum(abs, r_tsr - transpose(G) * r_abg) ≤
          1e3 * eps(Float64) * maximum(abs, transpose(G) * r_abg)
    # nothing is wrapped or clamped: the `:tsr` start's angles are the chart's own, unwrapped
    start = M.bubble_start(tsr, Float64)
    grid = M.bubble_start(abg, Float64)
    for k in 1:St
        α, β, γ = grid[k], grid[St + k], grid[2St + k]
        @test start[k] === atan(β, α)
        @test start[St + k] === -γ / sqrt(α^2 + β^2)
    end
end

# ---- FB16 ----------------------------------------------------------------------------

@testset "FB16 a failed step is carried and its record kept" begin
    # N22: a step that does not reach the tolerance leaves its last accepted iterate, the momentum
    # of that iterate and its three numbers; the next slab starts from them, and `on_failure` is
    # the one place that acts on the record.
    problem = clause_problem(:wave; timestep = 0.05, steps = 2)
    method = SpaceTimeGalerkin(clause_ansatz(problem), FixedBoundary();
        initial_guess = FitTo(Predictor()))
    int = GeometricIntegrator(problem, method; solver = LevenbergMarquardt(),
        max_iterations = 1, verbosity = 0)
    c = cache(int)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    reset!(solstep, timesteps(solution)[1])
    M.slab_bottom!(c, c.coupling, problem, ansatz(method), current(solstep))
    M.initial_guess!(current(solstep), nothing, parameters(problem), int)
    integrate!(solstep, int)
    level = copy(c.ansatz.params[1:M.nspatial(ansatz(method))])
    momentum = copy(c.coupling.p_next)
    @test !c.converged[end]
    @test c.iterations[end] == 1
    @test isfinite(c.residuals[end]) && isfinite(c.initial_residuals[end])
    # the next slab's datum is what the failed step left
    reset!(solstep, timesteps(solution)[2])
    M.slab_bottom!(c, c.coupling, problem, ansatz(method), current(solstep))
    @test c.coupling.phi == level
    @test c.coupling.momentum == momentum
    # the run completes and records both steps, and `on_failure` acts on the record. A failing step
    # warns once, so the run's log is exactly the two failures and no other record.
    result = integrate(problem, method; solver = LevenbergMarquardt(), max_iterations = 1,
        verbosity = 0)
    @test length(result.stats.converged) == 2
    @test !result.stats.converged[1]
    @test length(result.traces) == 2
    @test_logs (:warn, r"did not converge at timestep n=1") (:warn,
        r"did not converge at timestep n=2") integrate(
        problem, method; solver = LevenbergMarquardt(), max_iterations = 1, verbosity = 0,
        on_failure = :warn)
    @test_throws r"did not converge at timestep n=1" integrate(problem, method;
        solver = LevenbergMarquardt(), max_iterations = 1, verbosity = 0,
        on_failure = :error)
end
