# Whether two builds of one network architecture give the same kernels, and what that does to the
# trained fit of test E4a (known issue K59).
#
# Part 1 builds `NetworkAnsatz(S = 20, D = 1)` `builds` times, with a `GC.gc()` after every
# `gcevery`-th build, and sorts the builds into classes by the generated code of their two kernels
# and by the kernels' values at fixed inputs. One class means the build is reproducible in this
# process.
#
# Part 2 is E4a's pair, looped: the first step of a `PreviousSlab` run, `churn` builds of the same
# architecture and one `GC.gc()`, then the first step of a fresh `InitialCondition` integrator. It
# counts the trials whose residual or initial residual is not bitwise equal, and the largest relative
# difference. Where every integrator of one architecture shares one kernel pair, it counts none.
#
#     julia --startup-file=no --project=. scripts/kernel_build_order.jl [builds gcevery trials churn]
using MultisymplecticIntegrators, Random
using GeometricBase, GeometricIntegratorsBase, GeometricSolutions
import GeometricIntegratorsBase: cache, current, solutionstep, reset!
import SymbolicNeuralNetworks
const M = MultisymplecticIntegrators
const RGF = SymbolicNeuralNetworks.RuntimeGeneratedFunctions

builds, gcevery, trials, churn = length(ARGS) == 4 ? parse.(Int, ARGS) : (20, 5, 20, 3)

function build(T)
    NetworkAnsatz(S = 20, D = 1, mode = :last_layer, xspan = (zero(T), one(T)),
        degree = 3, ncells = (2, 5))
end

function code(a)
    string(RGF.get_expression(M.flat_kernel!(a.field))) *
    string(RGF.get_expression(M.flat_kernel!(a.gradient)))
end

function kernel_values(a, ::Type{T}) where {T}
    rng = Xoshiro(7)
    ps = M.unflatten(M.flat_layout(a.field), T.(2 .* rand(rng, a.np) .- 1))
    vals, grad = zeros(T, 3 * a.D), zeros(T, 3 * a.D, a.np)
    out = T[]
    for _ in 1:50
        input = T.([rand(rng), rand(rng)])
        M.flat_kernel!(a.field)(vals, input, ps, 1)
        M.flat_kernel!(a.gradient)(grad, input, ps, 1)
        append!(out, vals, vec(grad))
    end
    return out
end

reltol(::Type{Float64}) = 1.0e-2
reltol(::Type{Float32}) = 1.0e-1

function first_step(problem, ::Type{T}, target) where {T}
    method = SpaceTimeGalerkin(build(T), WeakInitialAndBoundaryConditions();
        initial_guess = FitTo(target; method = TrainedHiddenLayer(epochs = 20)))
    int = GeometricIntegrator(problem, method; f_abstol = zero(T), f_reltol = reltol(T),
        max_iterations = 2)
    solution = GeometricSolution(problem)
    step = solutionstep(int, solution[0])
    reset!(step, timesteps(solution)[1])
    M.integrate_step!(current(step), nothing, parameters(problem), int)
    return (cache(int).residuals[1], cache(int).initial_residuals[1])
end

function wave_without_solutions(::Type{T}) where {T}
    wave = Wave.problem(T = T, timespan = (zero(T), T(0.25)), timestep = T(0.05),
        xstep = T(0.05))
    return LagrangianPDEProblem(Wave.density(); D = 1, ics_function = wave.ics_function,
        bcs_function = wave.bcs_function, timespan = wave.timespan, timestep = wave.timestep,
        xspan = wave.xspan, xstep = wave.xstep, params = wave.params)
end

for T in (Float64, Float32)
    codes, values = String[], Vector{T}[]
    for k in 1:builds
        a = build(T)
        push!(codes, code(a))
        push!(values, kernel_values(a, T))
        gcevery > 0 && k % gcevery == 0 && GC.gc()
    end
    code_class = join(findfirst(==(c), unique(codes)) for c in codes)
    value_class = join(findfirst(v -> isequal(v, u), unique(values)) for u in values)
    spread = maximum(maximum(abs.(v .- values[1]) ./ max.(abs.(values[1]), eps(T)))
    for v in values)

    problem = wave_without_solutions(T)
    fails, worst = 0, zero(T)
    for _ in 1:trials
        a = first_step(problem, T, PreviousSlab())
        foreach(_ -> build(T), 1:churn)
        GC.gc()
        b = first_step(problem, T, InitialCondition())
        if a[1] !== b[1] || a[2] !== b[2]
            fails += 1
            worst = max(worst, abs(a[1] - b[1]) / abs(a[1]), abs(a[2] - b[2]) / abs(a[2]))
        end
    end
    @info("kernel builds and E4a's pair",
        T, builds, gcevery, code_class, value_class, spread,
        trials, churn, fails, worst, VERSION, check_bounds = Base.JLOptions().check_bounds)
end
