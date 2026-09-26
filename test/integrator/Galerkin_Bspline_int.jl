using Test
using GeometricIntegratorsBase
using GeometricSolutions
using MultiSymplectic

@testset "Full multiplier carries top time momentum to next slab" begin
    lpde = Wave.lpdeproblem(
        timestep = 0.05,
        timespan = (0.0, 0.1),
        xspan = (0.0, 1.0),
        xstep = 0.05
    )
    basis = BSpline2D(
        3;
        xspan = lpde.xspan,
        t_knot_interval = 0.5,
        x_knot_interval = 0.05
    )
    method = Galerkin_Bspline_Integrator(
        basis;
        xspan = lpde.xspan,
        RT_per_interval = 3,
        RX_per_interval = 3,
        show_status = false
    )
    int = MultiSymplectic.PDEIntegrator(lpde, method)
    sol = GeometricSolution(lpde)
    solstep = MultiSymplectic.solutionstep(int, sol[0])
    C = MultiSymplectic.cache(int)

    carried = GeometricIntegratorsBase.internal(solstep)
    @test !haskey(carried, :μ₀_t_coes)
    @test !haskey(carried, :μ₁_t_coes)

    carried.λ₁_x_coes .= reshape(1.0:length(C.λ₀_x_coes), size(C.λ₀_x_coes))
    MultiSymplectic.copy_internal_variables!(C, solstep)
    @test_broken C.λ₀_x_coes == carried.λ₁_x_coes    # issue #NN1

    C.λ₁_x_coes .= 2 .* carried.λ₁_x_coes
    @test_broken (C.λ₁_carry_x_coes .= 3 .* carried.λ₁_x_coes; true)    # issue #NN1
    C.ut₁_quad_values .= 3
    C.vt₁_quad_values .= 4
    C.wt₁_quad_values .= 5
    MultiSymplectic.copy_internal_variables!(solstep, C)

    @test_broken carried.λ₁_x_coes == C.λ₁_carry_x_coes    # issue #NN1
    @test_broken carried.λ₁_x_coes != C.λ₁_x_coes    # issue #NN1
    @test carried.ut₁_quad_values == C.ut₁_quad_values
    @test carried.vt₁_quad_values == C.vt₁_quad_values
    @test carried.wt₁_quad_values == C.wt₁_quad_values
end
