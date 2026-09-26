using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

# integrator/NN_PDE_int.jl is listed last because it redefines
# GeometricIntegratorsBase.default_options for NN_PDE_Integrator, which would otherwise change
# what any test after it measures.

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "Dirichlet multiplier momentum carry" include("integrator/Galerkin_Dirichlet_Bspline_int.jl")
    @safetestset "Full multiplier momentum carry" include("integrator/Galerkin_Bspline_int.jl")
    @safetestset "FEM multisymplectic wave integrator" include("integrator/FEM_Multisymplectic_int.jl")
    @safetestset "Space-time spline wave integrator" include("integrator/SpaceTime_Spline_Wave_int.jl")
    @safetestset "NN multiplier momentum carry" include("integrator/NN_PDE_int.jl")
end
