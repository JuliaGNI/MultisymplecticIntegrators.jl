using Test
using GeometricIntegratorsBase
using GeometricSolutions
using MultiSymplectic

@testset "NN multiplier carries top momentum to next slab" begin
    GeometricIntegratorsBase.default_options(::NN_PDE_Integrator) = (
        max_iterations = 1,
        warn_iterations = 1,
        verbosity = 0
    )

    lpde = Wave.lpdeproblem(
        timestep = 0.05,
        timespan = (0.0, 0.1),
        xspan = (0.0, 1.0),
        xstep = 0.05
    )
    basis = NetworkPDEBasis(2, tanh, :Partially)
    method = NN_PDE_Integrator(
        basis;
        xspan = lpde.xspan,
        RT_per_interval = 2,
        RX_per_interval = 2,
        t_num_interval = 2,
        x_num_interval = 2,
        Nw = 2,
        Nb = 2,
        show_status = false
    )
    int = MultiSymplectic.PDEIntegrator(lpde, method)
    sol = GeometricSolution(lpde)
    solstep = MultiSymplectic.solutionstep(int, sol[0])
    C = MultiSymplectic.cache(int)

    carried = GeometricIntegratorsBase.internal(solstep)
    @test haskey(carried, :λ₁_x_coes)

    carried.λ₁_x_coes .= reshape(1.0:length(C.λ₀_x_coes), size(C.λ₀_x_coes))
    MultiSymplectic.copy_internal_variables!(C, solstep)
    @test C.λ₀_x_coes == carried.λ₁_x_coes

    C.λ₁_x_coes .= 2 .* carried.λ₁_x_coes
    C.ut₁_quad_values .= 3
    C.vt₁_quad_values .= 4
    C.wt₁_quad_values .= 5
    MultiSymplectic.copy_internal_variables!(solstep, C)

    @test carried.λ₁_x_coes == C.λ₁_x_coes
    @test carried.ut₁_quad_values == C.ut₁_quad_values
    @test carried.vt₁_quad_values == C.vt₁_quad_values
    @test carried.wt₁_quad_values == C.wt₁_quad_values
end
