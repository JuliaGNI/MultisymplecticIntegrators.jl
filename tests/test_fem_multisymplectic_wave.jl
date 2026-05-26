using Test
using MultiSymplectic

@testset "FEM multisymplectic wave integrator" begin
    lpde = Wave.lpdeproblem(
        timestep = 0.05,
        timespan = (0.0, 200.0),
        xspan = (0.0, 1.0),
        xstep = 0.05
    )

    method = FEM_Multisymplectic_Integrator(lpde; startup = :exact)
    sol = MultiSymplectic.integrate(lpde, method)

    xs = collect(lpde.xspan[1]:lpde.xstep:lpde.xspan[2])
    err = maximum(abs.(sol.sol.u[end] .- lpde.exact_u.(lpde.timespan[2], xs)))

    hamiltonian_errors = zeros(length(lpde.timespan[1]:lpde.timestep:lpde.timespan[2]-lpde.timestep))
    for (i, t) in enumerate(lpde.timespan[1]:lpde.timestep:lpde.timespan[2]-lpde.timestep)
        numerical_hamiltonian = Wave.hamiltonian(
            sol.u_quad_values[i],
            sol.v_quad_values[i],
            sol.w_quad_values[i],
            method.grid_matrix,
            method.grid_weights,
            lpde.params,
            lpde.xspan,
            lpde.timestep,
        )

        exact_u = zeros(1, method.RT, method.RX)
        exact_v = zeros(1, method.RT, method.RX)
        exact_w = zeros(1, method.RT, method.RX)
        for rt in axes(method.grid_matrix, 1), rx in axes(method.grid_matrix, 2)
            exact_u[1, rt, rx] = lpde.exact_u(t + lpde.timestep * method.grid_matrix[rt, rx][1], method.grid_matrix[rt, rx][2]; params = lpde.params)
            exact_v[1, rt, rx] = lpde.exact_v(t + lpde.timestep * method.grid_matrix[rt, rx][1], method.grid_matrix[rt, rx][2]; params = lpde.params)
            exact_w[1, rt, rx] = lpde.exact_w(t + lpde.timestep * method.grid_matrix[rt, rx][1], method.grid_matrix[rt, rx][2]; params = lpde.params)
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

        hamiltonian_errors[i] = (numerical_hamiltonian - exact_hamiltonian) / exact_hamiltonian
    end

    @test err < 1e-2
    @test maximum(abs.(hamiltonian_errors)) < 1e-2
    @test size(sol.u_quad_values[end], 3) == length(xs)
    @test method.RT == 1
    @test method.RX == length(xs)
end
