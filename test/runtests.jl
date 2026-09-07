using Test

# Only the files in this directory that actually assert something are run here. The rest are
# numerical experiments -- they integrate for thousands of steps and write .jld2 archives and
# figures -- and are run by hand, not by the suite.
#
# test_spacetime_spline_wave_int.jl states its assertions at top level, so it is given the
# @testset it lacks. test_nn_multiplier_momentum_carry.jl is included last because it
# redefines GeometricIntegratorsBase.default_options for NN_PDE_Integrator, which would
# otherwise change what any test after it measures.

@testset "MultiSymplectic" begin
    include("test_dirichlet_multiplier_momentum_carry.jl")
    include("test_full_multiplier_momentum_carry.jl")
    include("test_fem_multisymplectic_wave.jl")

    @testset "space-time spline wave integrator" begin
        include("test_spacetime_spline_wave_int.jl")
    end

    include("test_nn_multiplier_momentum_carry.jl")
end
