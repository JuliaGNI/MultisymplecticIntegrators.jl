using Test
using MultisymplecticIntegrators

@testset "Galerkin_Full_Restriction_Bspline_Integrator shows its physical x-breaks" begin
    # The knot interval and the knots are in physical x, so neither is scaled by the width of
    # xspan: on `xspan = (0.0, 2.0)` a break of 0.5 prints as 0.5, not as 1.0.
    basis = Dirichlet_BSpline2D(
        3;
        timestep = 0.1,
        xspan = (0.0, 2.0),
        t_knot_interval = 0.5,
        x_knot_interval = 0.5
    )
    method = Galerkin_Full_Restriction_Bspline_Integrator(basis; xspan = (0.0, 2.0))
    shown = sprint(show, method)
    @test occursin("Break Length x: 0.5,", shown)
    @test occursin("\n   Breaks_x: [0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0]", shown)
end
