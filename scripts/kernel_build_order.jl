# Whether two symbolic builds of one network architecture give the same kernels, and whether test
# E4a's pair of first steps is bitwise equal (known issue K59).
#
# Part 1 builds the `field` and `gradient` kernels of the network `S = 20`, `D = 1`, `tanh`
# `builds` times through `network_functions`, around the per-process cache of `network_kernels`,
# with a `GC.gc()` after every `gcevery`-th build. It sorts the builds into classes by the generated
# code of the two kernels and by their values at fixed inputs. One class means that the symbolic
# build is reproducible in this process; more than one is the order dependence of K59.
#
# Part 2 is E4a's pair, looped: the first step of a `PreviousSlab` run, `churn` builds of the same
# `NetworkAnsatz` and one `GC.gc()`, then the first step of a fresh `InitialCondition` integrator. It
# counts the trials whose residual or initial residual is not bitwise equal, and the largest
# relative difference. Every `NetworkAnsatz` of one architecture reads the kernels of
# `network_kernels`, so it counts none wherever that cache holds.
#
#     julia --startup-file=no --project=. scripts/kernel_build_order.jl [builds gcevery trials churn]
using MultisymplecticIntegrators, Random
using GeometricBase, GeometricIntegratorsBase, GeometricSolutions
import GeometricIntegratorsBase: cache, current, solutionstep, reset!
import SymbolicNeuralNetworks
const M = MultisymplecticIntegrators
const RGF = SymbolicNeuralNetworks.RuntimeGeneratedFunctions

builds, gcevery, trials, churn = length(ARGS) == 4 ? parse.(Int, ARGS) : (20, 5, 20, 3)

const S, D = 20, 1

function build(T)
    NetworkAnsatz(S = S, D = D, mode = :last_layer, xspan = (zero(T), one(T)),
        degree = 3, ncells = (2, 5))
end

# A symbolic build of the kernels that does not read the cache.
function fresh_kernels()
    chain = M.Chain(M.Dense(2, S, tanh), M.Dense(S, D, identity; use_bias = false))
    return M.network_functions(M.SymbolicNeuralNetwork(chain), D)
end

function code(field, gradient)
    string(RGF.get_expression(M.flat_kernel!(field))) *
    string(RGF.get_expression(M.flat_kernel!(gradient)))
end

function kernel_values(field, gradient, np, ::Type{T}) where {T}
    rng = Xoshiro(7)
    ps = M.unflatten(M.flat_layout(field), T.(2 .* rand(rng, np) .- 1))
    vals, grad = zeros(T, 3 * D), zeros(T, 3 * D, np)
    out = T[]
    for _ in 1:50
        input = T.([rand(rng), rand(rng)])
        M.flat_kernel!(field)(vals, input, ps, 1)
        M.flat_kernel!(gradient)(grad, input, ps, 1)
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
    np = build(T).np
    codes, values = String[], Vector{T}[]
    for k in 1:builds
        field, gradient = fresh_kernels()
        push!(codes, code(field, gradient))
        push!(values, kernel_values(field, gradient, np, T))
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
