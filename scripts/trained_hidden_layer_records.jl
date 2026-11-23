# The records of the trained hidden layer that train for many epochs, which the suite does not run:
# the misfit fraction of E1a's fit after 60 epochs, and E5a's two section A soundings of the old
# stack's ELM formulation at 100 epochs. `test/initial_guess/trained_hidden_layer.jl` checks the
# training itself — one epoch descends, and the gradient it descends is the misfit's derivative.
#
#     julia --startup-file=no --project=scripts scripts/trained_hidden_layer_records.jl
#
# Each record is a testset; the script ends with an error if any of its assertions fails.
using GeometricBase: parameters, timesteps
using GeometricIntegratorsBase: GeometricIntegrator, cache, current, reset!, solutionstep
using GeometricSolutions: GeometricSolution
using LinearAlgebra: BLAS
using MultisymplecticIntegrators
using Test

const M = MultisymplecticIntegrators

include(joinpath(@__DIR__, "..", "test", "helpers", "reference_values.jl"))

"""The hidden units of the family's run."""
const TRAINED_S = 20

"""The hidden units of the fit that is measured on a misfit or on its sample."""
const FIT_S = 8

"""
The fraction of the drawn hidden layer's misfit that 60 epochs at `learning_rate = 1e-2` must
reach, measured on `arm64-apple-darwin27.0.0` at Julia 1.13.3: `5.55e-1` with the hidden layer drawn
and its output layer solved, and `4.41e-2` after the training, a ratio of `0.079`.
"""
const TRAINED_MISFIT_FRACTION = 1 // 2

"""
The two misfits of E1a's fit, by element type and components: `drawn` is the fit that solves the
output layer alone on the hidden layer `Xoshiro(1234)` draws, and `trained` the same fit after 60
epochs at `learning_rate = 1e-2`. Measured on `arm64-apple-darwin27.0.0` at Julia 1.13.3:
`0.5551 → 0.04409` in `Float64` at both `D = 1` and `D = 2`, `0.5551 → 0.04453` in `Float32` at
`D = 1`, and `0.5551 → 0.04431` in `Float32` at `D = 2`, where the least-squares solve carries a
second, zero column and its QR path differs. The same fits at Julia 1.12.7 leave `0.04407` in
`Float32` at `D = 1` and `0.04389` at `D = 2`.

The values are recorded, not asserted: a `Float32` misfit moves in its third significant digit
between the two Julia versions and under a one-ulp change of its learning rate, so no bound on it
holds on every leg. The script asserts that the drawn misfit is finite and positive, that the
trained one is at most `TRAINED_MISFIT_FRACTION` of the drawn one, and that the trained one reads
back from the network's own evaluation; it asserts neither value against its `const`.
"""
const TRAINED_MISFITS = (
    drawn = Dict{Tuple{DataType, Int}, Float64}(
        (Float64, 1) => 0.5551, (Float64, 2) => 0.5551,
        (Float32, 1) => 0.5551, (Float32, 2) => 0.5551),
    trained = Dict{Tuple{DataType, Int}, Float64}(
        (Float64, 1) => 0.04409, (Float64, 2) => 0.04409,
        (Float32, 1) => 0.04453, (Float32, 2) => 0.04431))

function wave_problem(::Type{T}; timestep = T(0.05), xstep = T(0.05),
        timespan = (zero(T), T(0.25))) where {T}
    return Wave.problem(T = T, timespan = timespan, timestep = timestep, xstep = xstep)
end

function sine_gordon_problem(::Type{T}; timestep = T(0.05)) where {T}
    return SineGordon.problem(T = T, timespan = (zero(T), T(0.15)), timestep = timestep,
        xstep = T(0.05))
end

"""A `VectorWave` problem on the vector density of the coupling `β`, which is `D = 2`. Its data
functions and its exact solution are `VectorWave.problem`'s."""
function vector_wave_problem(::Type{T}; β, timestep = T(0.05),
        timespan = (zero(T), T(0.5))) where {T}
    params = VectorWave.default_parameters()
    base = VectorWave.problem(T = T; params = params, timestep = timestep,
        timespan = timespan)
    return LagrangianPDEProblem(VectorWave.vector_density(; params = params, β = T(β));
        D = 2,
        ics_function = base.ics_function,
        bcs_function = base.bcs_function,
        solutions = base.solutions,
        timespan = timespan, timestep = timestep, xspan = base.xspan, xstep = base.xstep,
        params = params)
end

"""The method of one `:last_layer` network on `Wave` with the guess it is given."""
function trained_method(::Type{T}; S = FIT_S, D = 1, guess) where {T}
    return SpaceTimeGalerkin(
        NetworkAnsatz(S = S, D = D, mode = :last_layer, xspan = (zero(T), one(T)),
            degree = 3, ncells = (2, 5)),
        WeakInitialAndBoundaryConditions(); initial_guess = guess)
end

"""The first slab of a run, set up as the family runs it: its datum in the coupling's block, and
the solver's vector from the method's guess, before the solver runs."""
function trained_slab(problem, method)
    int = GeometricIntegrator(problem, method)
    solution = GeometricSolution(problem)
    step = solutionstep(int, solution[0])
    reset!(step, timesteps(solution)[1])
    c = cache(int)
    M.slab_bottom!(c, c.coupling, problem, ansatz(method), current(step))
    M.initial_guess!(step, nothing, parameters(problem), int)
    return int
end

"""The misfit of a fitted parameter vector on the fit's own sample, `Σ_k ω_k (u_k − f_k)²`, read
back through the network's own evaluation."""
function fitted_misfit(block, w)
    f = block.train
    copyto!(block.params, w)
    total = zero(eltype(w))
    for k in axes(f.points, 2)
        M.evaluate_at!(block, f.points[1, k], f.points[2, k])
        for d in axes(f.target, 2)
            total += f.measure[k] * abs2(block.values[d] - f.target[k, d])
        end
    end
    return total
end

"""The sup-norm deviation of a run from the `u` of a §A entry, over the steps it took."""
function sup_deviation(result, entry)
    return maximum(maximum(abs.(result.solution.u[n] .- entry.u[n, :]))
    for n in 1:length(result.traces))
end

"""Whether every state of a solution series is finite."""
finite_states(x) = all(n -> all(isfinite, x[n]), eachindex(x))

"""
    record_deviation(record, entry, T, deviation)

Write one reference deviation with the metadata that makes the figure readable on another machine.
No bound is asserted on the value.
"""
function record_deviation(record::Symbol, entry::Symbol, ::Type{T}, deviation) where {T}
    @info("the reference deviation of an ELM run", record, entry, T, deviation, VERSION,
        machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
end

"""
    record_misfit(record, T, D, misfit, recorded)

Write one misfit of a fit on its own sample with the metadata that makes the figure readable on
another machine, beside the `const` the same fit is recorded as. The value is not compared with its
`const`: a `Float32` misfit moves in its third significant digit under a one-ulp change of its
learning rate, so the script asserts the fraction between the two misfits of one fit and reads the
trained one back through the network's own evaluation, and compares neither with its `const`.
"""
function record_misfit(record::Symbol, ::Type{T}, D, misfit, recorded) where {T}
    @info("the misfit of a trained fit on its own sample", record, T, D, misfit, recorded,
        VERSION, machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
end

@testset "Trained hidden layer records" begin
    # E1a's fit at its two ends: the drawn hidden layer with its output layer solved, and the same
    # after 60 epochs of training, on a scalar field and on both components of a vector field.
    @testset "the misfit fraction after 60 epochs" for T in (Float64, Float32), D in (1, 2)

        scalar(t, x, p) = sin(2 * T(π) * x) * cos(T(π) * t)
        target = D === 1 ? scalar : (t, x, p) -> (scalar(t, x, p), zero(T))
        problem = D === 1 ? wave_problem(T) :
                  vector_wave_problem(T; β = zero(T), timespan = (zero(T), T(0.25)))
        make(epochs, lr) = trained_method(T; S = FIT_S, D = D,
            guess = FitTo(TargetFunction(target);
                method = TrainedHiddenLayer(epochs = epochs, learning_rate = lr)))
        drawn = cache(trained_slab(problem, make(0, T(1e-2)))).ansatz
        trained = cache(trained_slab(problem, make(60, T(1e-2)))).ansatz
        @test isfinite(drawn.train.misfit)
        @test isfinite(trained.train.misfit)
        @test drawn.train.misfit > zero(T)
        record_misfit(:drawn, T, D, drawn.train.misfit, TRAINED_MISFITS.drawn[(T, D)])
        record_misfit(:trained, T, D, trained.train.misfit, TRAINED_MISFITS.trained[(T, D)])
        @test trained.train.misfit ≤
              convert(T, TRAINED_MISFIT_FRACTION) * drawn.train.misfit
        # the misfit read back through the network's own evaluation is the fit's
        @test fitted_misfit(trained, copy(trained.params)) ≈ trained.train.misfit rtol = sqrt(eps(T))
        # both fits were taken on the same sample
        @test drawn.train.target == trained.train.target
        @test drawn.train.measure == trained.train.measure
    end

    # The two records of the old stack's ELM formulation, at its own cells: a run of each completes
    # and its deviation from the entry is written with its metadata. Both entries are unconverged
    # iterates of another architecture, so the deviation is a record (K44).
    @testset "E5a the two section A soundings" for T in (Float64,)
        elm_method(problem) = SpaceTimeGalerkin(
            NetworkAnsatz(S = TRAINED_S, mode = :last_layer, xspan = problem.xspan,
                ncells = (10, 10), degree = 3),
            WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition();
                method = TrainedHiddenLayer(epochs = 100, learning_rate = T(1e-5))))
        cases = ((:WAVE_ELM_LSGD, wave_problem(T), WAVE_ELM_LSGD, 5),
            (:SINE_GORDON_ELM_LSGD, sine_gordon_problem(T), SINE_GORDON_ELM_LSGD, 3))
        for (name, problem, entry, steps) in cases
            int = GeometricIntegrator(problem, elm_method(problem))
            result = integrate(int)
            c = cache(int)
            @test length(result.traces) == steps
            @test length(c.iterations) == steps
            @test finite_states(result.solution.u)
            deviation = sup_deviation(result, entry)
            @test isfinite(deviation)
            record_deviation(:E5a, name, T, deviation)
            @info("the ELM record", T, name, c.iterations, c.residuals,
                c.initial_residuals, c.converged, deviation)
        end
    end
end
