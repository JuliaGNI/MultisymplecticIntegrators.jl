using Test
using MultiSymplectic

# @testset "space-time spline wave variational integrator" begin
    lpde = Wave.lpdeproblem(
        timestep = 0.05,
        timespan = (0.0, 5.0),
        xspan = (0.0, 1.0),
        xstep = 0.05,
    )
    basis = Dirichlet_BSpline2D(
        3;
        timestep = lpde.timestep,
        xspan = lpde.xspan,
        t_knot_interval = 0.5,
        x_knot_interval = 0.05,
    )
    method = SpaceTime_Spline_Wave_Integrator(basis, lpde)
    sol = MultiSymplectic.integrate(lpde, method)

    xs = collect(lpde.xspan[1]:lpde.xstep:lpde.xspan[2])
    err = maximum(abs.(sol.sol.u[end] .- lpde.exact_u.(lpde.timespan[2], xs)))

    hamiltonian_errors = zeros(length(lpde.timespan[1]:lpde.timestep:lpde.timespan[2]-lpde.timestep))
    exact_u = zeros(1, method.RT, method.RX)
    exact_v = zeros(1, method.RT, method.RX)
    exact_w = zeros(1, method.RT, method.RX)
    for (step, t) in enumerate(lpde.timespan[1]:lpde.timestep:lpde.timespan[2]-lpde.timestep)
        numerical_hamiltonian = Wave.hamiltonian(
            sol.u_quad_values[step],
            sol.v_quad_values[step],
            sol.w_quad_values[step],
            method.grid_matrix,
            method.grid_weights,
            lpde.params,
            lpde.xspan,
            lpde.timestep,
        )
        for rt in axes(method.grid_matrix, 1), rx in axes(method.grid_matrix, 2)
            τ, x = method.grid_matrix[rt, rx]
            exact_u[1, rt, rx] = lpde.exact_u(t + lpde.timestep * τ, x; params=lpde.params)
            exact_v[1, rt, rx] = lpde.exact_v(t + lpde.timestep * τ, x; params=lpde.params)
            exact_w[1, rt, rx] = lpde.exact_w(t + lpde.timestep * τ, x; params=lpde.params)
        end
        exact_hamiltonian = Wave.hamiltonian(
            exact_u,
            exact_v,
            exact_w,
            method.grid_matrix,
            method.grid_weights,
            lpde.params,
            lpde.xspan,
            lpde.timestep,
        )
        hamiltonian_errors[step] = (numerical_hamiltonian - exact_hamiltonian) / exact_hamiltonian
    end

    @test err < 1e-3
    @test maximum(abs.(hamiltonian_errors)) < 1e-3
    @test method.RT > 1
    @test method.RX > length(xs)
# end
