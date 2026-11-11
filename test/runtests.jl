using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "Explicit imports" include("quality/explicit_imports.jl")
    @safetestset "Sources and docstrings" include("quality/sources.jl")
    @safetestset "Lagrangian density" include("problems/lagrangian_density.jl")
    @safetestset "Lagrangian problem" include("problems/lagrangian_problem.jl")
    @safetestset "Linear transport" include("problems/linear_transport.jl")
    @safetestset "Vector wave" include("problems/vector_wave.jl")
    @safetestset "Tensor B-spline ansatz" include("ansatz/tensor_bspline.jl")
    @safetestset "Coefficient restriction" include("couplings/coefficient_restriction.jl")
    @safetestset "Coupling seam" include("core/slab_integrator.jl")
    @safetestset "Initial guess" include("initial_guess/guess.jl")
    @safetestset "Integrator shell" include("integration/integrator_shell.jl")
    @safetestset "Archives" include("integration/archives.jl")
    @safetestset "Registry" include("integration/registry.jl")
    @safetestset "Diagnostics" include("diagnostics.jl")
    @safetestset "Plotting" include("plots.jl")
end

if "slow" in GROUPS
    @safetestset "Symbolic ansatz" include("ansatz/symbolic.jl")
    @safetestset "Weak multipliers" include("couplings/weak_multipliers.jl")
    @safetestset "Network ansatz" include("ansatz/network.jl")
    @safetestset "Orthogonal greedy" include("initial_guess/orthogonal_greedy.jl")
    @safetestset "Trained hidden layer" include("initial_guess/trained_hidden_layer.jl")
    @safetestset "PINN training" include("initial_guess/pinn_training.jl")
    @safetestset "Hard constraint lifting" include("couplings/hard_constraint_lifting.jl")
    @safetestset "Property: generated problems" include("verification/properties/generator.jl")
    @safetestset "Property: gradient" include("verification/properties/gradient.jl")
    @safetestset "Property: Jacobian" include("verification/properties/jacobian.jl")
    @safetestset "Property: linear control" include("verification/properties/linear_control.jl")
    @safetestset "Property: Noether charge" include("verification/properties/noether.jl")
    @safetestset "Property: symmetry" include("verification/properties/symmetry.jl")
    @safetestset "Fixed boundary" include("couplings/fixed_boundary.jl")
end
