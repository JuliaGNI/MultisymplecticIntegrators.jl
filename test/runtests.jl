using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "Explicit imports" include("quality/explicit_imports.jl")
    @safetestset "Lagrangian density" include("problems/lagrangian_density.jl")
    @safetestset "Lagrangian problem" include("problems/lagrangian_problem.jl")
    @safetestset "Linear transport" include("problems/linear_transport.jl")
    @safetestset "Vector wave" include("problems/vector_wave.jl")
    @safetestset "Space-time quadrature" include("quadrature/spacetime_quadrature.jl")
    @safetestset "Tensor B-spline ansatz" include("ansatz/tensor_bspline.jl")
    @safetestset "Coefficient restriction" include("couplings/coefficient_restriction.jl")
    @safetestset "Weak multipliers" include("couplings/weak_multipliers.jl")
    @safetestset "Initial guess" include("initial_guess/guess.jl")
    @safetestset "Integrator shell" include("integration/integrator_shell.jl")
end
