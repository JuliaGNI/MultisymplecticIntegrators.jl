# The fixed-boundary coupling: the two types and their pair; the momentum, the side data, the solves
# and their tolerances over one or two steps; the runs, the continuation, the predictor, the bubble
# fit, the element type, the two charts and a failed step; the family's rows. One of the files of
# `test/couplings/fixed_boundary*.jl`, which share `test/helpers/fixed_boundary_setup.jl`.
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

# ---- FB0 -----------------------------------------------------------------------------

@testset "FB0 the two types and their pair" begin
    problem = PROBLEMS[:wave]()
    a = FixedBoundaryNetwork(Sx = 4, St = 4, xspan = (0.0, 1.0))
    @test FixedBoundary() isa SlabCoupling
    @test a isa SpaceTimeAnsatz
    @test (M.nspatial(a), M.nbubble(a), M.nfree(a)) == (12, 16, 28)
    last_layer = FixedBoundaryNetwork(Sx = 4, St = 4, xspan = (0.0, 1.0), mode = :last_layer)
    @test (M.nspatial(last_layer), length(M.spatial_free(last_layer)),
        M.nfree(last_layer)) == (12, 4, 8)
    # At `D = 2` a level frees `Sx·D` output weights, which are not the first `Sx` of them: the
    # level's parameters are `Sx` weights, `Sx` biases and then `Sx·D` output weights.
    wide = FixedBoundaryNetwork(
        Sx = 8, St = 10, xspan = (0.0, 1.0), mode = :last_layer, D = 2)
    @test M.nspatial(wide) == 2 * 8 + 8 * 2
    @test M.spatial_free(wide) == (2 * 8 + 1):(2 * 8 + 8 * 2)
    @test M.bubble_free(wide) == (3 * 10 + 1):(3 * 10 + 10 * 2)
    @test M.nfree(wide) == 8 * 2 + 10 * 2
    @test M.free_indices(wide) == vcat(collect(17:32), 32 .+ collect(31:50))
    network = NetworkAnsatz(S = 4, xspan = (0.0, 1.0), ncells = (2, 5), degree = 3)
    spline = TensorBSplineAnsatz(degree = 3, ncells = (2, 5), xspan = (0.0, 1.0))
    # `@test_throws` matches its `Regex` against `sprint(showerror, e)`, which prefixes the
    # exception's type, so the anchors hold only when the type is part of the pattern.
    @test_throws r"^ArgumentError: FixedBoundary takes a FixedBoundaryNetwork, got NetworkAnsatz$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(network, FixedBoundary()))
    @test_throws r"^ArgumentError: FixedBoundary takes a FixedBoundaryNetwork, got TensorBSplineAnsatz$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(spline, FixedBoundary()))
    for coupling in (WeakInitialAndBoundaryConditions(), WeakInitialCondition(),
        CoefficientRestriction())
        @test_throws Regex("^ArgumentError: a FixedBoundaryNetwork takes the FixedBoundary " *
                           "coupling, got $(nameof(typeof(coupling)))\$") GeometricIntegrator(
            problem, SpaceTimeGalerkin(a, coupling))
    end
    # `HardConstraintLifting` takes a `NetworkAnsatz` or a `TensorBSplineAnsatz` and names those two
    # in its own refusal; a `FixedBoundaryNetwork` is neither, so the refusal that answers is this
    # intersection's.
    @test_throws r"^ArgumentError: a FixedBoundaryNetwork takes the FixedBoundary coupling, got HardConstraintLifting$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(a, HardConstraintLifting()))
    method = SpaceTimeGalerkin(a, FixedBoundary())
    int = GeometricIntegrator(problem, method)
    @test default_solver(method) isa Newton
    @test solver(int) isa SimpleSolvers.NonlinearSolver{Float64, <:Newton}
    # Levenberg–Marquardt is the solver a caller names, and the family builds it as its own
    named = GeometricIntegrator(problem, method; solver = LevenbergMarquardt())
    @test solver(named) isa M.LevenbergMarquardtSolver
    # the `FitTo(Predictor())` guess changes the guess and not the solver
    predicted = SpaceTimeGalerkin(a, FixedBoundary(); initial_guess = FitTo(Predictor()))
    @test solver(GeometricIntegrator(problem, predicted)) isa
          SimpleSolvers.NonlinearSolver{Float64, <:Newton}
    @test length(cache(int).x) == 28
    # A `PINNTraining` fit on a fixed-boundary network matches the network's refusal for any fit
    # method and `PINNTraining`'s refusal for any ansatz, so it reaches this intersection's.
    @test_throws r"^ArgumentError: a FixedBoundaryNetwork is fitted by BubbleGreedy, got PINNTraining$" M.fit!(
        zeros(1), PINNTraining(epochs = 0), a, nothing, cache(int))
    @test M.coupling_unknowns(FixedBoundary(), a, problem) == 0
    @test M.free_rows(FixedBoundary(), a) == 28
    @test total_free_coefficients(method, problem) == 28
    # A `PreviousSlab` target has no field map for this ansatz, so the guess is refused and the
    # refusal names the target it was given.
    @test_throws r"^ArgumentError: a FixedBoundaryNetwork takes InitialCondition, ExactSolution, TargetFunction and Predictor as targets, got PreviousSlab$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(a, FixedBoundary(); initial_guess = FitTo(PreviousSlab())))
end

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
        FixedBoundary())) isa Newton
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
    # The default `Newton()` factorizes with LAPACK's pivoted QR, which refuses an element type
    # LAPACK does not provide, so a caller at such a type names the solver as well as the tolerance.
    # Such a caller runs: the option reaches the solver's config unmoved, and the same build without
    # the tolerance raises the same refusal.
    big = WF.problem(T = BigFloat, timespan = (big"0.0", big"0.1"), timestep = big"0.05",
        xstep = big"0.05", xspan = (big"0.0", big"1.0"))
    big_method = SpaceTimeGalerkin(fixed_ansatz(big; Sx = 4, St = 4), FixedBoundary())
    # the refusal's text is SimpleSolvers', so only its type is this package's to assert
    # ast-grep-ignore: test-throws-bare-type
    @test_throws ArgumentError GeometricIntegrator(big, big_method; f_reltol = big"1e-30")
    big_int = GeometricIntegrator(big, big_method; solver = LevenbergMarquardt(),
        f_reltol = big"1e-30")
    @test eltype(nlsolution(cache(big_int))) === BigFloat
    @test config(solver(big_int)).f_reltol == big"1e-30"
    @test_throws r"^ArgumentError: the fixed-boundary network has no measured f_reltol at BigFloat; name the tolerance as an option of its own$" GeometricIntegrator(
        big, big_method; solver = LevenbergMarquardt())
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
    # ... and so is a caller that names neither the tolerance nor the solver: the family's default
    # solver resolves the tolerance as every other one does
    @test_throws r"^ArgumentError: the fixed-boundary network has no measured f_reltol at BigFloat; name the tolerance as an option of its own$" GeometricIntegrator(
        big, big_method)
end

# ---- FB9 -----------------------------------------------------------------------------

@testset "FB9 the dead branch on Wave over twenty steps" begin
    problem = WF.problem(timespan = (0.0, 1.0), timestep = 0.05, xstep = 0.05,
        xspan = (0.0, 1.0))
    ansatz = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = :abg, xspan = (0.0, 1.0),
        ncells = 20, nq = (6, 4))
    # the family's defaults: the guess, the solver, its rank cut, its tolerance and its cap
    method = SpaceTimeGalerkin(ansatz, FixedBoundary())
    result = integrate(problem, method; verbosity = 0)
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
    method32 = SpaceTimeGalerkin(ansatz32, FixedBoundary())
    result32 = integrate(problem32, method32; verbosity = 0)
    @info("FB9 the dead branch's Float32 arm", VERSION, T="Float32",
        ratios=result32.stats.residuals ./ result32.stats.initial_residuals,
        converged=result32.stats.converged,
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
    # The workspace keys on the method converted to the slab's element type, so its field is
    # concretely typed, and a `Float32` slab keeps a `Float32` key and `Float32` atoms.
    for T in (Float64, Float32)
        @test fieldtype(M.FixedBoundaryFit{T}, :method) ===
              Union{Nothing, BubbleGreedy{NTuple{2, T}}}
        @test isconcretetype(BubbleGreedy{NTuple{2, T}})
    end
    p32 = WF.problem(T = Float32, timespan = (0.0f0, 0.1f0), timestep = 0.05f0,
        xstep = 0.05f0, xspan = (0.0f0, 1.0f0))
    a32 = fixed_ansatz(p32)
    s32 = fixed_slab(p32, a32)
    c32 = cache(s32.int)
    tb32 = a32.tabulation
    sample32 = zeros(Float32, length(tb32.tnode), length(tb32.xnode), a32.D)
    M.evaluate!(sample32, InitialCondition(), c32, a32, p32, current(s32.solstep))
    w32 = collect(c32.ansatz.params)
    M.fit!(w32, BubbleGreedy(nα = 3, nβ = 3, nγ = 3), a32, sample32, c32)
    @test c32.ansatz.fit.method isa BubbleGreedy{NTuple{2, Float32}}
    @test c32.ansatz.fit.atoms isa Matrix{Float32}
    # The dictionary is built from ends converted to the element type: at `BigFloat` the value
    # `-1/3` of the range `(-1, 1)` on four points carries `BigFloat`'s precision, not `Float64`'s,
    # in each of the three columns.
    third = bubble_atoms(
        BubbleGreedy(nα = 4, nβ = 4, nγ = 4, α_range = (-1.0, 1.0), β_range = (-1.0, 1.0),
            γ_range = (-1.0, 1.0)),
        BigFloat)
    for j in 1:3
        @test minimum(abs(third[i, j] + big(1) / 3) for i in axes(third, 1)) ≤
              4 * eps(BigFloat)
    end
    # A `BigFloat` slab converts the `Float64` ranges to new `BigFloat` objects at every fit; the
    # key compares by value, so a second fit with the same method keeps the table the first built,
    # and a method of other ranges builds it again. `GeometricSolutions` builds no solution at
    # `BigFloat` (K53) and the fits of the levels take an `svd` that `BigFloat` lacks, so the slab
    # gets its side data alone and the bubble's fit runs on its own.
    pbig = WF.problem(T = BigFloat, timespan = (big"0.0", big"0.1"), timestep = big"0.05",
        xstep = big"0.05", xspan = (big"0.0", big"1.0"))
    abig = fixed_ansatz(pbig; Sx = 4, St = 4)
    cbig = cache(GeometricIntegrator(pbig, SpaceTimeGalerkin(abig, FixedBoundary());
        solver = LevenbergMarquardt(), f_reltol = big"1e-30"))
    tbbig = abig.tabulation
    for (row, t̂) in enumerate((zero(BigFloat), tbbig.tnode..., one(BigFloat)))
        M.side_row!(cbig.coupling, pbig, row, timestep(pbig) * t̂)
    end
    samplebig = BigFloat[sin(π * x) * (1 + t)
                         for t in tbbig.tnode, x in tbbig.xnode, _ in 1:abig.D]
    wbig, small = collect(cbig.ansatz.params), BubbleGreedy(nα = 3, nβ = 3, nγ = 3)
    M.fit_bubble!(wbig, small, abig, cbig.coupling, samplebig, cbig.ansatz)
    table = cbig.ansatz.fit.table
    @test cbig.ansatz.fit.method isa BubbleGreedy{NTuple{2, BigFloat}}
    M.fit_bubble!(wbig, small, abig, cbig.coupling, samplebig, cbig.ansatz)
    @test cbig.ansatz.fit.table === table
    # each of the three ranges is part of the key
    for other in ((; α_range = (-4.0, 4.0)), (; β_range = (-4.0, 4.0)),
        (; γ_range = (-3.0, 3.0)))
        M.fit_bubble!(wbig, small, abig, cbig.coupling, samplebig, cbig.ansatz)
        before = cbig.ansatz.fit.table
        M.fit_bubble!(wbig, BubbleGreedy(; nα = 3, nβ = 3, nγ = 3, other...), abig,
            cbig.coupling, samplebig, cbig.ansatz)
        @test cbig.ansatz.fit.table !== before
    end
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

# ---- FB18 ----------------------------------------------------------------------------

@testset "FB18 the family's rows build and the plateau is a record" begin
    # N23: the registry holds the family's 32 entries — 30 rows of the matrix and 2 runs of the
    # plateau probe — each with a distinct stem, one of the plateau's at `Sx = 12`, `St = 16` and
    # `h = 0.025` and one at `BigFloat`, and the plateau's numbers are a record and no assertion.
    plateau = family_entries(:fixed_boundary)
    matrix = family_entries(:fixed_boundary_matrix)
    @test length(plateau) == 2
    @test length(matrix) == 30
    @test allunique(entry.stem for entry in Iterators.flatten((plateau, matrix)))
    @test all(entry -> occursin("plateau", entry.stem), plateau)
    @test !any(entry -> occursin("plateau", entry.stem), matrix)
    @test all(entry -> entry.steps ≥ 1, plateau)
    @test all(entry -> entry.parameters.steps ≥ 1, matrix)
    @test length(Set((entry.problem, entry.parameters.form, entry.parameters.solve,
                         entry.resolution.timestep) for entry in matrix)) == 30
    # a matrix row integrates its ladder up to `t = 0.2`, the final time of the other families'
    # rows of its problem, so that the rows of one problem compare errors at one time
    @test all(entry -> entry.parameters.steps * entry.resolution.timestep ≈ 0.2, matrix)
    @test sort(unique(entry.parameters.steps for entry in matrix)) == [2, 4, 8]
    # a driver given no stem runs the plateau's `Float64` entry, and not the `BigFloat` one, by its
    # stem and whatever the order of the family's tuple: reversed, the cheapest entry of the tie is
    # the `BigFloat` one, and the default stays
    default = default_entry(:fixed_boundary)
    @test default.stem == "wave-fixed-boundary-plateau-h0.025"
    @test default.T === Float64
    held = FAMILIES[:fixed_boundary]
    try
        FAMILIES[:fixed_boundary] = merge(held, (entries = reverse(held.entries),))
        @test smallest_entry(family_entries(:fixed_boundary)).T === BigFloat
        @test default_entry(:fixed_boundary) === default
    finally
        FAMILIES[:fixed_boundary] = held
    end
    # a plateau entry names its solve as a row's `parameters` do; `solver` is a row's solver method
    @test all(entry -> entry.solve in (:newton, :joint, :projected), plateau)
    @test !any(entry -> haskey(entry, :solver), plateau)
    # the cheapest row, the largest time step, runs at `Float64`
    largest = maximum(entry.resolution.timestep for entry in matrix)
    @test all(entry -> entry.parameters.T === :Float64,
        filter(entry -> entry.resolution.timestep == largest, matrix))
    entries = vcat(collect(plateau), collect(matrix))
    @test all(entry -> entry.problem in (:wave, :sine_gordon), entries)
    @test any(entry -> entry.Sx == 12 && entry.St == 16 && entry.timestep == 0.025, plateau)
    @test any(entry -> entry.T === BigFloat, plateau)
    @test all(entry -> entry.steps == 1, filter(entry -> entry.T === BigFloat, plateau))
    # N23: the extended-precision entry names its own tolerance, because the family measures none at
    # `BigFloat`; the entry's driver and this testset build the solver's keywords through one
    # function, so the entry's value is the one the solver is built with
    extended = only(filter(entry -> entry.T === BigFloat, plateau))
    @test extended.f_reltol == big"1e-8"
    extended_int = GeometricIntegrator(fixed_boundary_problem(extended),
        fixed_boundary_method(extended); solver = M.LevenbergMarquardt(),
        fixed_boundary_solver_keywords(extended)...)
    @test config(solver(extended_int)).f_reltol == extended.f_reltol
    for entry in entries
        @test fixed_boundary_problem(entry) isa M.LagrangianPDEProblem
        @test fixed_boundary_method(entry) isa SpaceTimeGalerkin
    end
end

# ---- FB19 ----------------------------------------------------------------------------

@testset "FB19 the defaults converge on the matrix's rows" begin
    # The matrix's rows at `h = 0.1` and `h = 0.05`, both problems and both bubble forms, under the
    # family's defaults alone: `Continuation()`, `Newton()` on the pivoted QR cut at
    # `fixed_boundary_qr_cut`, `f_reltol = fixed_boundary_f_reltol`, `f_abstol = 0` and 300
    # iterations. Every step converges; the worst ratio and the error are a record.
    for P in (M.Wave, M.SineGordon), h in (0.1, 0.05), form in (:abg, :tsr)
        problem = P.problem(timespan = (0.0, 0.2), timestep = h, xstep = h,
            xspan = (0.0, 1.0))
        ansatz = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = form,
            xspan = (0.0, 1.0), ncells = 20, nq = (6, 4))
        result = integrate(problem, SpaceTimeGalerkin(ansatz, FixedBoundary()); verbosity = 0)
        stats = result.stats
        t = result.solution.t
        u = result.solution.u
        xs = collect(xgrid(problem))
        error = maximum(abs(P.solution_u(t[n], x, parameters(problem)) - u[n][i])
        for n in 1:ntime(problem), (i, x) in enumerate(xs))
        @info("FB19 a row under the defaults", problem=nameof(P), h, form,
            worst_ratio=maximum(stats.residuals ./ stats.initial_residuals),
            iterations=stats.iterations, error, cpu=Sys.CPU_NAME)
        @testset "$(nameof(P)), h = $h, $form" begin
            @test length(stats.converged) == (h == 0.1 ? 2 : 4)
            @test all(stats.converged)
        end
    end
end
