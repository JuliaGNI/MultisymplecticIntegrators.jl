# The shared setup of the network ansatz test files `test/ansatz/network*.jl`: the §A case's
# constants, its problem, run and control, the first slab of a run, and the deviations, differences
# and Gauss rule the files compare against. Each file includes it after its own `using` lines and
# the helpers it names.

const M = MultisymplecticIntegrators

# The deviation of a network run from its §A entry is a record, not an assertion: at `S = 10` the
# `:last_layer` runs have sixteen constraint rows that reach only ten output weights, and the Newton
# solve of the `:all` runs does not reach round-off (K38), so every step of these runs is an
# unconverged Newton iterate, and no bound on its distance from the entry holds across machines.
# Each test asserts that the deviation is finite and writes it with `@info`, beside the machine that
# produced it; K38 and K39 record the figures of one machine.
#
# Neither entry reproduces bit for bit: the code that wrote them reads the bottom points in the
# normalized coordinate (`xspan` is `(0, 1)`, so there the two coincide) and keeps a duplicate atom
# in its dictionary, which costs the greedy a whole selection.

"""The steps of the §A case."""
const NETWORK_STEPS = 5

"""The most Newton iterations the affine control of a `:last_layer` step may take."""
const LAST_LAYER_ITERATIONS = 2

"""The hidden units of the affine control, which frees its output weights alone."""
const NETWORK_CONTROL_S = 20

function network_ansatz(::Type{T}; S = 10, D = 1, mode = :all, degree = 3, ncells = (2, 5),
        xspan = (zero(T), one(T)), quadrature_order = nothing) where {T}
    return NetworkAnsatz(S = S, D = D, mode = mode, xspan = xspan, degree = degree,
        ncells = ncells, quadrature_order = quadrature_order)
end

function wave_problem(::Type{T}; timespan = (zero(T), T(0.25)), timestep = T(0.05),
        xstep = T(0.05), D = 1) where {T}
    return Wave.problem(
        T = T, D = D, timespan = timespan, timestep = timestep, xstep = xstep)
end

"""A whole run of one network mode on `Wave`: five steps of `h = 0.05`."""
function network_run(::Type{T}; mode = :all, D = 1, S = 10,
        guess = FitTo(InitialCondition(); method = OrthogonalGreedy()),
        problem = wave_problem(T; D = D), steps = NETWORK_STEPS) where {T}
    method = SpaceTimeGalerkin(network_ansatz(T; mode = mode, D = D, S = S),
        WeakInitialAndBoundaryConditions(); initial_guess = guess)
    int = GeometricIntegrator(problem, method)
    result = integrate(int)
    return problem, method, int, result
end

"""The sup-norm deviation of a run from the `u` of a §A entry, over the steps it took."""
function sup_deviation(result, entry)
    return maximum(maximum(abs.(result.solution.u[n] .- entry.u[n, :]))
    for n in 1:length(result.traces))
end

"""Whether every state of a solution series is finite."""
function finite_states(x)
    return all(n -> all(isfinite, x[n]), eachindex(x))
end

"""
    record_deviation(record, entry, T, deviation)

Write one §A deviation with the metadata that makes the figure readable on another machine: the
testset that measured it, the entry it is a distance from, the element type, the Julia version, the
machine, the CPU, the BLAS thread count and the `--check-bounds` setting of the process (`0` auto,
`1` yes, `2` no). No bound is asserted on the value.
"""
function record_deviation(record::Symbol, entry::Symbol, ::Type{T}, deviation) where {T}
    @info("the §A deviation of a network run", record, entry, T, deviation, VERSION,
        machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(),
        check_bounds=Base.JLOptions().check_bounds)
end

"""Whether two solution series are bit-identical on every state they hold."""
function same_states(x, y)
    return axes(x) == axes(y) && all(n -> x[n] == y[n], eachindex(x))
end

"""
    network_control(::Type{T})

The affine control of a `:last_layer` slab: `S = 20` hidden units, whose freed output weights are
the only unknowns the step's rows read, solved with `f_abstol = 0` at the family's relative
tolerance and at most two Newton iterations.
"""
function network_control(::Type{T}; steps = NETWORK_STEPS) where {T}
    problem = wave_problem(T; timespan = (zero(T), T(steps) * T(0.05)))
    method = SpaceTimeGalerkin(
        network_ansatz(T; S = NETWORK_CONTROL_S, mode = :last_layer),
        WeakInitialAndBoundaryConditions();
        initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
    int = GeometricIntegrator(problem, method; f_abstol = zero(T),
        f_reltol = network_f_reltol(T), max_iterations = LAST_LAYER_ITERATIONS)
    return integrate(int)
end

"""
    network_slab(::Type{T}; S, mode, D, xspan, problem)

The first slab of a network run, set up as the property suite does: its datum in the coupling's
block, and the solver's vector from the method's guess, before the solver runs.
"""
function network_slab(::Type{T}; S = 10, mode = :all, D = 1, xspan = (zero(T), one(T)),
        problem = wave_problem(T; D = D)) where {T}
    method = SpaceTimeGalerkin(network_ansatz(T; S = S, mode = mode, D = D, xspan = xspan),
        WeakInitialAndBoundaryConditions();
        initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
    int = GeometricIntegrator(problem, method)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    reset!(solstep, timesteps(solution)[1])
    c = cache(int)
    M.slab_bottom!(c, c.coupling, problem, ansatz(method), current(solstep))
    M.initial_guess!(solstep, nothing, parameters(problem), int)
    return (; problem, method, int, solstep)
end

"""
    central_differences(int, θ, params)

The central differences of the slab's `residual!` at `θ`, one column per unknown, with the step
`δ_k = cbrt(eps(T)) max(1, |θ_k|)`. Each column divides by the step as it is represented.
"""
function central_differences(int, θ::AbstractVector{T}, params) where {T}
    N = length(θ)
    J = zeros(T, N, N)
    forward = zeros(T, N)
    backward = zeros(T, N)
    x = copy(θ)
    for k in 1:N
        δ = cbrt(eps(T)) * max(one(T), abs(θ[k]))
        x[k] = θ[k] + δ
        M.residual!(forward, x, nothing, params, int)
        up = x[k]
        x[k] = θ[k] - δ
        M.residual!(backward, x, nothing, params, int)
        J[:, k] .= (forward .- backward) ./ (up - x[k])
        x[k] = θ[k]
    end
    return J
end

# The physical Gauss nodes and weights of one direction of a slab, computed here so that a node set
# read in another coordinate shows up as a difference rather than as an error.
function gauss_rule(::Type{T}, a, b, ncells::Integer, n::Integer = 4) where {T}
    ξ = T[-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
    ω = T[0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
    nodes = T[]
    weights = T[]
    width = (b - a) / ncells
    for c in 0:(ncells - 1), k in 1:n

        push!(nodes, a + width * (T(c) + (ξ[k] + one(T)) / 2))
        push!(weights, width / 2 * ω[k])
    end
    return nodes, weights
end
