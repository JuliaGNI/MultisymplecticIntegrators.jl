using ParallelTestRunner
using MultisymplecticIntegrators

const TESTS = [
    "core" => [
        "Lagrangian density" => "problems/lagrangian_density.jl",
        "Lagrangian problem" => "problems/lagrangian_problem.jl",
        "Linear transport" => "problems/linear_transport.jl",
        "Vector wave" => "problems/vector_wave.jl",
        "Tensor B-spline ansatz" => "ansatz/tensor_bspline.jl",
        "Coefficient restriction" => "couplings/coefficient_restriction.jl",
        "Coupling seam" => "core/slab_integrator.jl",
        "Initial guess" => "initial_guess/guess.jl",
        "Integrator shell" => "integration/integrator_shell.jl",
        "Archives" => "integration/archives.jl",
        "Registry" => "integration/registry.jl",
        "Box scheme" => "baselines/box_scheme.jl",
        "Diagnostics" => "diagnostics.jl"
    ],
    "quality" => [
        "Aqua" => "quality/aqua.jl",
        "Explicit imports" => "quality/explicit_imports.jl",
        "Sources and docstrings" => "quality/sources.jl",
        "Inference" => "quality/inference.jl",
        "JET" => "quality/jet.jl",
        "Interface conformance" => "quality/interface.jl",
        "Plotting" => "plots.jl"
    ],
    "slow" => [
        "Benchmark matrix" => "integration/benchmark.jl",
    ],
    "ansatz" => [
        "Symbolic ansatz" => "ansatz/symbolic.jl",
        "Network ansatz" => "ansatz/network.jl"
    ],
    "couplings" => [
        "Weak multipliers" => "couplings/weak_multipliers.jl",
        "Hard constraint lifting" => "couplings/hard_constraint_lifting.jl"
    ],
    "initialguess" => [
        "Orthogonal greedy" => "initial_guess/orthogonal_greedy.jl",
        "Trained hidden layer" => "initial_guess/trained_hidden_layer.jl",
        "PINN training" => "initial_guess/pinn_training.jl"
    ],
    "verification" => [
        "Property: generated problems" => "verification/properties/generator.jl",
        "Property: gradient" => "verification/properties/gradient.jl",
        "Property: Jacobian" => "verification/properties/jacobian.jl",
        "Property: linear control" => "verification/properties/linear_control.jl",
        "Property: Noether charge" => "verification/properties/noether.jl",
        "Property: symmetry" => "verification/properties/symmetry.jl",
        "Manufactured solutions" => "verification/manufactured.jl",
        "Long-time energy" => "verification/long_time_energy.jl"
    ],
    "fixedboundary" => [
        "Fixed boundary" => "couplings/fixed_boundary.jl",
    ]
]

const SELECTED = filter(!startswith("--"), ARGS)
const GROUPS = isempty(SELECTED) ?
               ["core", "quality", "slow", "ansatz", "couplings", "initialguess",
    "verification", "fixedboundary"] :
               SELECTED
const UNKNOWN = setdiff(SELECTED, first.(TESTS))
isempty(UNKNOWN) ||
    error("no test group ", join(UNKNOWN, ", "), " in runtests.jl; its groups are ",
        join(first.(TESTS), ", "))

runtests(MultisymplecticIntegrators, filter(startswith("--"), ARGS);
    testsuite = Dict(label => :(include($(joinpath(@__DIR__, path))))
    for (group, files) in TESTS if group in GROUPS for (label, path) in files),
    serial = [label for (group, files) in TESTS if group in ("metal", "cuda")
              for (label, _) in files],
    serial_position = :after,
    retries = 0,
    exeflags = ["--threads=$(Threads.nthreads())"],
    max_default_jobs = max(1, Sys.CPU_THREADS ÷ Threads.nthreads()))
