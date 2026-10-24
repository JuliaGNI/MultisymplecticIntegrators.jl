using Test
using MultisymplecticIntegrators

@testset "Dirichlet_BSpline2D shows its physical x-breaks" begin
    # On xspan = (0, 2) the width is 2, so a break scaled by the width once more shows twice its
    # value: `Break Length x: 1.0` and `[0.0, 0.0, 0.0, 1.0, …]`.
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
