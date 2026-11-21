# The fixed-boundary coupling: the momentum, the side data, the solves and their tolerances over one
# or two steps. One of the files of `test/couplings/fixed_boundary*.jl`, which share
# `test/helpers/fixed_boundary_setup.jl`.
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

# ---- FB3 -----------------------------------------------------------------------------

@testset "FB3 the first momentum and the momentum handed on" begin
    # N6: the datum of a run's first slab — the fit of the level to `u₀`, and `p₀` against the
    # oracle's own integral of the density's momentum, on the free parameters of the mode
    for name in (:wave, :sinegordon), mode in (:all, :last_layer)

        problem = clause_problem(name)
        ansatz = clause_ansatz(problem; mode = mode)
        slab = fixed_slab(problem, ansatz)
        c = cache(slab.int)
        p = parameters(problem)
        # the fit's sup error on the output grid, whose 21 points are `x = 0:0.05:1`
        xs = collect(range(xspan(problem)[1], xspan(problem)[2]; length = 21))
        initial = problem.ics_function(xs, p)
        fitted = [level_network(problem, ansatz, c.coupling.phi, x, 0.0, d)
                  for x in xs, d in 1:ansatz.D]
        @test maximum(abs, fitted .- initial.u) ≤ 1e-5
        # the covector of the mode's own parameters: `Sx·D` under `:last_layer`
        @test length(c.coupling.momentum) == length(M.spatial_free(ansatz))
        nodes, weights = ansatz.tabulation.xnode, ansatz.tabulation.xweight
        ics = problem.ics_function(collect(nodes), p)
        expect = zeros(length(c.coupling.momentum))
        for r in eachindex(nodes)
            g = ForwardDiff.gradient(
                φ -> sum(level_network(problem, ansatz, φ, nodes[r],
                             0.0, d) for d in 1:ansatz.D),
                c.coupling.phi)
            for (k, i) in enumerate(M.spatial_free(ansatz)), d in 1:ansatz.D

                expect[k] += weights[r] *
                             derivative_v(problem.density, ics.u[r, d],
                                 ics.v[r, d], ics.w[r, d], p) * g[i]
            end
        end
        @test all(isfinite, expect)
        @test maximum(abs, c.coupling.momentum - expect) ≤
              100 * eps(Float64) * maximum(abs, expect)
    end
    # N7: the momentum handed on, at the final iterate of a step that is capped at three LM
    # iterations and of one `Newton()` takes, unchanged by a further residual evaluation and by a
    # second step
    for solvermethod in (LevenbergMarquardt(), Newton())
        problem = clause_problem(:wave; steps = 2)
        ansatz = clause_ansatz(problem)
        method = SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = Continuation())
        int = GeometricIntegrator(problem, method; solver = solvermethod,
            max_iterations = 3, verbosity = 0)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        c = cache(int)
        p = parameters(problem)
        # A step's own time is the end of the slab it solves — `slab_start` reads one timestep back
        # from it — so a run's first step is placed at `timesteps(solution)[1]`, the first slab's
        # end, as the helper `next_fixed_slab!` places it.
        reset!(solstep, timesteps(solution)[1])
        M.slab_bottom!(c, c.coupling, problem, ansatz, current(solstep))
        datum = fixed_datum(c)
        oracle = ActionOracle(problem, method; block = c.ansatz)
        integrate!(solstep, int)
        solved = copy(c.x)
        handed = copy(c.coupling.p_next)
        solved_level = copy(c.ansatz.params[1:M.nspatial(ansatz)])
        scale = maximum(abs, handed)
        @test all(isfinite, handed)
        @test maximum(abs,
            oracle_momentum(oracle, datum, solved) - handed) ≤ 100 * eps(Float64) * scale
        # a further residual evaluation at another vector rewrites no part of the record — the
        # block's own parameters it does rewrite, which is why the level is read before it
        other = 2 .* rand(MersenneTwister(11), Float64, length(solved)) .- 1
        b = zeros(length(solved))
        M.residual!(b, other, nothing, p, int)
        @test c.coupling.p_next == handed
        # the second slab's datum is that block and that level, exactly
        next_fixed_slab!((; problem, method, int, solution, solstep, ansatz), 2)
        @test c.coupling.momentum == handed
        @test c.coupling.phi == solved_level
    end
end

# ---- FB4 -----------------------------------------------------------------------------

@testset "FB4 the solution carries the side data at every step" begin
    # The state is component-major: component `d`'s `nx` output-grid values start at
    # `(d - 1) * nx + 1`, and the grid's ends are the span's.
    function ends(u, D)
        nx = length(u) ÷ D
        return [(u[(d - 1) * nx + 1], u[d * nx]) for d in 1:D]
    end
    for name in (:wave, :sinegordon), form in (:abg, :tsr)

        problem = clause_problem(name; steps = 3)
        method = SpaceTimeGalerkin(clause_ansatz(problem; form = form), FixedBoundary())
        result = integrate(problem, method; max_iterations = 3, verbosity = 0)
        @test length(result.traces) == 3
        for n in eachindex(result.stats.converged)
            t = result.solution.t[n]
            bcs = problem.bcs_function(t, parameters(problem))
            scale = max(1, maximum(abs, result.solution.u[n]))
            for (d, (left, right)) in enumerate(ends(result.solution.u[n],
                components(problem)))
                @test abs(left - bcs.left.u[d]) ≤ 8 * eps(Float64) * scale
                @test abs(right - bcs.right.u[d]) ≤ 8 * eps(Float64) * scale
            end
        end
    end
    # The suite's three `:general` entries and one `:symmetric` entry, at the clause's 20 cells: the
    # identity holds at every node of any rule, and the two kinds carry different side data — a
    # `:symmetric` entry's two ends agree and vary in time, a `:general` entry's differ.
    entries = generate_problems(Float64; rng = Xoshiro(SEED))
    legs = vcat([(entry = entries[i], kind = :general) for i in (13, 14, 15)],
        [(entry = entries[9], kind = :symmetric)])
    for leg in legs, form in (:abg, :tsr)

        @test leg.entry.kind === leg.kind
        problem = leg.entry.problem
        method = SpaceTimeGalerkin(clause_ansatz(problem; form = form), FixedBoundary())
        result = integrate(problem, method; max_iterations = 3, verbosity = 0)
        for n in eachindex(result.stats.converged)
            bcs = problem.bcs_function(result.solution.t[n], parameters(problem))
            scale = max(1, maximum(abs, result.solution.u[n]))
            for (d, (left, right)) in enumerate(ends(result.solution.u[n],
                components(problem)))
                @test abs(left - bcs.left.u[d]) ≤ 8 * eps(Float64) * scale
                @test abs(right - bcs.right.u[d]) ≤ 8 * eps(Float64) * scale
            end
        end
    end
end

# ---- FB5 -----------------------------------------------------------------------------

@testset "FB5 variable projection" begin
    for name in (:wave, :sinegordon), form in (:abg, :tsr)

        problem = name === :wave ?
                  WF.problem(timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05,
            xspan = (0.0, 1.0)) :
                  SG.problem(timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05,
            xspan = (0.0, 1.0))
        ansatz = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = form,
            xspan = (0.0, 1.0), ncells = 20, nq = (6, 4), D = components(problem))
        slab = fixed_slab(problem, ansatz; guess = FitTo(Predictor()),
            solver = LevenbergMarquardt(variable_projection = true), max_iterations = 5,
            verbosity = 0)
        int, c = slab.int, cache(slab.int)
        p = parameters(problem)
        slot = length(M.variable_projection_indices(slab.method))
        @test slot == ansatz.St * ansatz.D
        @test M.variable_projection_indices(slab.method) ==
              collect((M.nfree(ansatz) - slot + 1):M.nfree(ansatz))
        projected = M.variable_projection_indices(slab.method)
        n = length(c.x)
        oracle = ActionOracle(problem, slab.method; block = c.ansatz)
        datum = fixed_datum(c)
        M.initial_guess!(current(slab.solstep), nothing, p, int)
        θ₀ = copy(c.x)
        # A fixed-boundary guess is a function of the slab's datum alone, so a second application of
        # it writes the same vector; that is what lets the two applications a step makes agree.
        M.initial_guess!(current(slab.solstep), nothing, p, int)
        @test c.x == θ₀
        f! = (b, z) -> (M.residual!(b, z, nothing, p, int); b)
        # 1. the inner solve: after it, the eliminated block is at a root of its own equations, and
        # the block was not already there — both at the oracle's gradient
        g₀ = oracle_residual(oracle, datum, θ₀)
        scale = maximum(abs, view(g₀, projected))
        @test scale ≥ sqrt(eps(Float64))
        x = copy(θ₀)
        r = slab_residual(int, x, p)
        reduced, _ = M.project!(x, f!, r, M.jacobian_of(f!, x, n), projected)
        floor = maximum(abs, view(oracle_residual(oracle, datum, x), projected))
        # the same rule as FB1's, at the rows FB5 measures: the inner solve's residual is compared
        # with the larger of the old bound and the round-off floor of the oracle's own gradient at
        # this point, read at the projected rows
        noise = oracle_floor(oracle, datum, x; rows = projected)
        @test noise ≤ 1e6 * eps(Float64) * scale
        @test floor ≤ max(100 * eps(Float64) * scale, noise)
        @info("FB5 the inner solve",
            VERSION, T="Float64", name=name, form=form, scale=scale,
            floor=floor, noise=noise, machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
        @info("FB5 the inner solve's floor", VERSION, T="Float64", name=name, form=form,
            ratio=Float64(noise/(eps(Float64)*scale)),
            machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(),
            check_bounds=Base.JLOptions().check_bounds)
        # 2. the Schur complement against central differences of z ↦ F_o(z, a*(z)), each inner solve
        # warm-started at the solved `a`
        solved = M.lm_solved_indices(n, projected)
        function reduced_residual(z)
            rp = slab_residual(int, z, p)
            M.project!(copy(z), f!, rp, M.jacobian_of(f!, z, n), projected)
            return view(rp, solved)
        end
        z = copy(x)
        worst = zero(Float64)
        for (k, j) in enumerate(solved)
            δ = cbrt(eps(Float64)) / 10 * max(1, abs(x[j]))
            up = copy(z)
            up[j] += δ
            down = copy(z)
            down[j] -= δ
            difference = (reduced_residual(up) .- reduced_residual(down)) ./ (2δ)
            ratio = maximum(abs, difference - view(reduced, :, k)) /
                    (sqrt(eps(Float64)) * maximum(abs, reduced))
            worst = max(worst, ratio)
            @test ratio ≤ 1
        end
        @info("FB5 the reduced Jacobian against differences",
            VERSION, T="Float64", name=name, form=form,
            worst=worst, machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
        # 3. one projected step: its recorded residual is the test's own full-system `‖F‖`
        guess = c.x
        r₀ = norm(slab_residual(int, guess, p))
        integrate!(slab.solstep, int)
        x₁ = copy(c.x)
        @test c.iterations[end] ≤ 5
        @test abs(c.residuals[end] - norm(slab_residual(int, x₁, p))) ≤
              8 * eps(Float64) * max(1, c.residuals[end])
        @test abs(c.initial_residuals[end] - r₀) ≤ 8 * eps(Float64) * max(1, r₀)
    end
end

# ---- FB6 -----------------------------------------------------------------------------

@testset "FB6 the record of a step under each form and solve" begin
    @test default_solver(SpaceTimeGalerkin(fixed_ansatz(PROBLEMS[:wave]()),
        FixedBoundary())) isa LevenbergMarquardt
    for name in (:wave, :sinegordon), form in (:abg, :tsr),
        solve in (:lm, :projected, :newton)
        problem = PROBLEMS[name]()
        a = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = form,
            xspan = xspan(problem), ncells = 20, nq = (6, 4), D = components(problem))
        m = SpaceTimeGalerkin(a, FixedBoundary(); initial_guess = FitTo(Predictor()))
        solvermethod = if solve === :lm
            LevenbergMarquardt()
        elseif solve === :projected
            LevenbergMarquardt(variable_projection = true)
        else
            Newton()
        end
        int = GeometricIntegrator(problem, m; solver = solvermethod, max_iterations = 3,
            verbosity = 0)
        c = cache(int)
        p = parameters(problem)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        options = config(solver(int))
        ns = M.nspatial(a)
        solved = Float64[]
        for n in 1:2
            reset!(solstep, timesteps(solution)[n])
            M.slab_bottom!(c, c.coupling, problem, a, current(solstep))
            # The datum of a later slab is the level the step before it solved, and not whatever
            # the guess of that step left in the block's parameter vector: a run under `integrate!`
            # applies the guess twice, and the second application overwrote that vector.
            n == 2 && @test c.coupling.phi == solved
            # A fixed-boundary guess is a function of the slab's datum alone — the fit and the
            # predictor read the datum, not the block's parameter vector — so the vector measured
            # here is the one the step's own guess writes; FB5 asserts that at the first slab.
            M.initial_guess!(current(solstep), nothing, p, int)
            r₀ = norm(slab_residual(int, c.x, p))
            integrate!(solstep, int)
            r₁ = norm(slab_residual(int, c.x, p))
            @test length(c.iterations) == n
            @test c.iterations[end] ≤ 3
            # the record against the test's own full-system norms, at the guess and at the final one
            @test abs(c.residuals[end] - r₁) ≤ 8 * eps(Float64) * max(1, r₁)
            @test abs(c.initial_residuals[end] - r₀) ≤ 8 * eps(Float64) * max(1, r₀)
            @test isfinite(c.residuals[end]) && isfinite(c.initial_residuals[end])
            @test c.converged[end] ==
                  M.step_converged(c.residuals[end], c.initial_residuals[end], options)
            solved = copy(c.ansatz.params[1:ns])
        end
    end
end

# ---- FB7 -----------------------------------------------------------------------------

@testset "FB7 the LM routine against a closed form" begin
    f!(b, x) = (b .= atan.(x))
    x = [2.0, -3.0]
    result = M.lm_solve!(x, f!, 2; max_iterations = 300, f_abstol = 0.0, f_reltol = 1e-12)
    @test result.converged
    @test norm(x) ≤ 1e-8
    @test result.residual == norm(atan.(x))
    @test result.initial_residual > 1
    y = [NaN, 0.0]
    nan = M.lm_solve!(y, f!, 2; max_iterations = 300, f_abstol = 0.0, f_reltol = 1e-8)
    @test nan.iterations == 0
    @test !nan.converged
end

# ---- FB8 -----------------------------------------------------------------------------

@testset "FB8 the affine control in two iterations" begin
    # N13 and N18's third check: the plan's control is `:last_layer` on a linear density, with the
    # spatial hidden layer held at `φ₀`'s fit for the whole run — which is the `Continuation()`
    # guess — and the unknowns `c` and `a`, in both forms and in both element types.
    # `FitTo(Predictor())` is not the control: it moves level 1's hidden layer away from level 0's,
    # and the affine system then has no solution (K50).
    #
    # The slab's Jacobian is rank-deficient, and the part of the initial residual outside its range is
    # a floor that no step removes: on entry 10 of the generated family it is `1.51e-10` of the
    # initial residual at the first slab, and one, two, three and five iterations all end there. The
    # floor is a property of the slab's data, and it moves with the machine's round-off. So each step
    # is held to the larger of the tolerance and ten times its own floor, which a dense pivoted QR of
    # the step's Jacobian measures outside the solver, and the step converges wherever that floor is
    # a tenth of the tolerance or less. The floor itself is bounded by `√f_reltol`: the largest on
    # these slabs is `1.29e-9` at `Float64` and `1.63e-2` at `Float32`, and a residual that has lost
    # its dependence on the unknowns has floor `1`, which would otherwise pass every check here.
    for T in (Float64, Float32), form in (:abg, :tsr)

        linear = [entry.problem
                  for entry in generate_problems(T; rng = Xoshiro(SEED))
                  if entry.potential == 2]
        for (i, problem) in enumerate(vcat([clause_problem(:wave; T = T)], linear))
            ansatz = clause_ansatz(problem; mode = :last_layer, form = form)
            method = SpaceTimeGalerkin(ansatz, FixedBoundary();
                initial_guess = Continuation())
            int = GeometricIntegrator(
                problem, method; solver = Newton(), max_iterations = 2,
                f_reltol = fixed_boundary_f_reltol(T), verbosity = 0)
            c = cache(int)
            p = parameters(problem)
            solution = GeometricSolution(problem)
            solstep = solutionstep(int, solution[0])
            floors = T[]
            for n in 1:ntime(problem)
                reset!(solstep, timesteps(solution)[n])
                M.slab_bottom!(c, c.coupling, problem, ansatz, current(solstep))
                M.initial_guess!(current(solstep), nothing, p, int)
                push!(floors, residual_floor(int, c.x, p, M.fixed_boundary_qr_cut(T)))
                integrate!(solstep, int)
            end
            ratios = c.residuals ./ c.initial_residuals
            reltol = fixed_boundary_f_reltol(T)
            label = i == 1 ? "wave" : "entry $(i - 1)"
            @info("FB8 the affine control's ratio and floor per step", label, T, form,
                ratios, floors, cpu = Sys.CPU_NAME)
            @testset "$label, $T, $form" begin
                @test length(ratios) == ntime(problem)
                @test all(c.iterations .≤ 2)
                @test all(floors .≤ sqrt(reltol))
                @test all(ratios .≤ max.(reltol, 10 .* floors))
                @test all(c.converged .| (floors .> reltol / 10))
                @test all(c.initial_residuals .≥ T(1 // 100))
            end
        end
    end
    @test fixed_boundary_f_reltol(Float64) == 1e-8
    @test fixed_boundary_f_reltol(Float32) ≥ 1e-8
    @test fixed_boundary_f_reltol(Float32) != Float32(1e-8)
    # N14: a third element type is refused rather than given a tolerance the family never measured,
    # and the refusal names it — matched whole and anchored, as every refusal of this file is.
    @test_throws r"^ArgumentError: the fixed-boundary network has no measured f_reltol at BigFloat; name the tolerance as an option of its own$" fixed_boundary_f_reltol(
        BigFloat)
    # A caller who names the tolerance at such a type runs: the option reaches the solver's config
    # unmoved, and the same build without it raises the same refusal.
    big = WF.problem(T = BigFloat, timespan = (big"0.0", big"0.1"), timestep = big"0.05",
        xstep = big"0.05", xspan = (big"0.0", big"1.0"))
    big_method = SpaceTimeGalerkin(fixed_ansatz(big; Sx = 4, St = 4), FixedBoundary())
    big_int = GeometricIntegrator(big, big_method; f_reltol = big"1e-30")
    @test eltype(nlsolution(cache(big_int))) === BigFloat
    @test config(solver(big_int)).f_reltol == big"1e-30"
    @test_throws r"^ArgumentError: the fixed-boundary network has no measured f_reltol at BigFloat; name the tolerance as an option of its own$" GeometricIntegrator(
        big, big_method)
    # the family's rank cut is one value per element type, `1e-12` at `Float64` and `10 eps(T)`
    # for another type
    @test M.fixed_boundary_qr_cut(Float64) == 1e-12
    @test M.fixed_boundary_qr_cut(Float32) == 10 * eps(Float32)
    problem = PROBLEMS[:wave]()
    int = GeometricIntegrator(problem,
        SpaceTimeGalerkin(fixed_ansatz(problem), FixedBoundary()))
    options = config(solver(int))
    @test options.f_abstol == 0
    @test options.f_reltol == fixed_boundary_f_reltol(Float64)
    @test options.max_iterations == 300
end

# ---- N14 -----------------------------------------------------------------------------

@testset "N14 the family's tolerance reaches every solver" begin
    # N14: `default_options` hands the tolerance on unresolved, and the family's `initsolver` fills
    # it in — the one method every nonlinear solver of the family reaches, because
    # `GeometricIntegrator` merges `default_options` before the caller's keywords. A solver that
    # names no tolerance of its own is solved at the element type's measured value, whichever
    # nonlinear solver it is.
    for T in (Float64, Float32), s in (SimpleSolvers.DogLeg(), SimpleSolvers.Picard())

        problem = clause_problem(:wave; T = T)
        int = GeometricIntegrator(problem,
            SpaceTimeGalerkin(fixed_ansatz(problem), FixedBoundary());
            solver = s, max_iterations = 5)
        @test config(solver(int)).f_reltol == fixed_boundary_f_reltol(T)
    end
    # at an element type the family has not measured, a caller that names the tolerance keeps it ...
    big = WF.problem(T = BigFloat, timespan = (big"0.0", big"0.1"), timestep = big"0.05",
        xstep = big"0.05", xspan = (big"0.0", big"1.0"))
    big_method = SpaceTimeGalerkin(fixed_ansatz(big; Sx = 4, St = 4), FixedBoundary())
    named = GeometricIntegrator(big, big_method; solver = SimpleSolvers.DogLeg(),
        f_reltol = big"1e-30")
    @test config(solver(named)).f_reltol == big"1e-30"
    # ... and a caller that names none is refused, whole and anchored, before a solver is built
    @test_throws r"^ArgumentError: the fixed-boundary network has no measured f_reltol at BigFloat; name the tolerance as an option of its own$" GeometricIntegrator(
        big, big_method; solver = SimpleSolvers.DogLeg())
end
