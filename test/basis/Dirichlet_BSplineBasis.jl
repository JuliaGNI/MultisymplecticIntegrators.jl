using Test
using MultisymplecticIntegrators

@testset "Dirichlet_BSpline2D shows its physical x-breaks" begin
    # The knot interval and the knots are in physical x, so neither is scaled by the width of
    # xspan: on `xspan = (0.0, 2.0)` a break of 0.5 prints as 0.5, not as 1.0.
    basis = Dirichlet_BSpline2D(
        3;
        timestep = 0.1,
        xspan = (0.0, 2.0),
        t_knot_interval = 0.5,
        x_knot_interval = 0.5
    )
    shown = sprint(show, basis)
    @test occursin("Break Length x: 0.5,", shown)
    @test occursin("Scaled Breaks_x: [0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0]", shown)
end
