module TrialNNTopLiftingExperiment

using MultiSymplectic
using GeometricIntegratorsBase: current, history, parameters, reset!

const MS = MultiSymplectic

"""
    previous_top_lifting(tn, x, int)

Value of the previous slab's lifting term at its top edge. For the current
`C1 - C2` construction this is the linear interpolation of the exact boundary
values at physical time `tn`.
"""
function previous_top_lifting(tn, x, int)
    a, b = int.problem.xspan
    xd = b - a
    exact_u = int.problem.exact_u

    return (b - x) * exact_u(tn, a) / xd +
           (x - a) * exact_u(tn, b) / xd
end

function previous_top_lifting_x(tn, int)
    a, b = int.problem.xspan
    xd = b - a
    exact_u = int.problem.exact_u

    return (exact_u(tn, b) - exact_u(tn, a)) / xd
end

"""
    initialize_bcs_ics_top_lifting!(sol, int)

Experimental replacement for `MultiSymplectic.initialize_bcs_ics!` for
`TrialNN_PDE_int`.

The only intentional difference is the history carry-over:

    old: current_C1C2(t, x) += previous_C1C2(t, x)
    new: current_C1C2(t, x) += (1 - t) * previous_C1C2_top(x)
"""
function initialize_bcs_ics_top_lifting!(sol, int)
    C = MS.cache(int)
    N = int.method.initial_guess_method.N
    quad_nodes = int.method.initial_guess_method.equispaced_quad_nodes
    tn = sol.t - MS.timestep(int)
    D = int.problem.D
    grid_matrix = int.method.grid_matrix
    RT = int.method.RT
    RX = int.method.RX
    h = MS.timestep(int)
    x_nodes = C.x_nodes

    C1C2_result = C.C1C2_result
    Ct_result = C.∂C1C2∂t_result
    Cx_result = C.∂C1C2∂x_result
    C1C2_equispaced = C.C1C2_equispaced_quad_nodes
    C1C2_quad = C.C1C2_quad
    Ct_quad = C.∂C1C2∂t_quad
    Cx_quad = C.∂C1C2∂x_quad

    for i in 1:N
        t = quad_nodes[1, i]
        x = quad_nodes[2, i]
        C1C2_equispaced[i] = MS.C1C2(t, x, tn, int, sol)
        if !iszero(tn)
            C1C2_equispaced[i] += (1 - t) * previous_top_lifting(tn, x, int)
        end
    end

    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                t, x = grid_matrix[rt, rx]
                C1C2_quad[d, rt, rx] = MS.C1C2(t, x, tn, int, sol)
                Ct_quad[d, rt, rx] = MS.∂C1C2∂t(t, x, tn, int, sol) / h
                Cx_quad[d, rt, rx] = MS.∂C1C2∂x(t, x, tn, int, sol)

                if !iszero(tn)
                    top = previous_top_lifting(tn, x, int)
                    C1C2_quad[d, rt, rx] += (1 - t) * top
                    Ct_quad[d, rt, rx] += -top / h
                    Cx_quad[d, rt, rx] += (1 - t) * previous_top_lifting_x(tn, int)
                end
            end
        end
    end

    for d in 1:D
        for i in eachindex(x_nodes)
            x = x_nodes[i]
            C1C2_result[d, i] = MS.C1C2(1.0, x, tn, int, sol)
            Ct_result[d, i] = MS.∂C1C2∂t(1.0, x, tn, int, sol) / h
            Cx_result[d, i] = MS.∂C1C2∂x(1.0, x, tn, int, sol)

            if !iszero(tn)
                Ct_result[d, i] += -previous_top_lifting(tn, x, int) / h
            end
        end
    end

    return C
end

function compare_current_and_top_lifting!(sol, int)
    C = MS.cache(int)

    MS.initialize_bcs_ics!(sol, int)
    old_equispaced = copy(C.C1C2_equispaced_quad_nodes)
    old_quad = copy(C.C1C2_quad)
    old_Ct_quad = copy(C.∂C1C2∂t_quad)
    old_Cx_quad = copy(C.∂C1C2∂x_quad)
    old_result = copy(C.C1C2_result)
    old_Ct_result = copy(C.∂C1C2∂t_result)
    old_Cx_result = copy(C.∂C1C2∂x_result)

    initialize_bcs_ics_top_lifting!(sol, int)

    return (
        equispaced = maximum(abs.(C.C1C2_equispaced_quad_nodes .- old_equispaced)),
        quad = maximum(abs.(C.C1C2_quad .- old_quad)),
        ∂t_quad = maximum(abs.(C.∂C1C2∂t_quad .- old_Ct_quad)),
        ∂x_quad = maximum(abs.(C.∂C1C2∂x_quad .- old_Cx_quad)),
        result = maximum(abs.(C.C1C2_result .- old_result)),
        ∂t_result = maximum(abs.(C.∂C1C2∂t_result .- old_Ct_result)),
        ∂x_result = maximum(abs.(C.∂C1C2∂x_result .- old_Cx_result)),
    )
end

function integrate_top_lifting!(solstep, int)
    reset!(solstep, MS.timestep(int))

    MS.copy_internal_variables!(MS.cache(int), solstep)
    initialize_bcs_ics_top_lifting!(solstep, int)
    MS.prior_initial_guess!(MS.cache(int), solstep, int)
    MS.integrate_step!(current(solstep), history(solstep), parameters(solstep), int)
    MS.components!(MS.nlsolution(int), current(solstep), parameters(solstep), int)
    MS.copy_internal_variables!(solstep, MS.cache(int))

    if MS.cache(int).flag_done_initial_guess[1] != 0.0
        MS.cache(int).flag_done_initial_guess[1] = 0.0
    end

    return solstep
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using GeometricIntegratorsBase
    using GeometricSolutions
    using MultiSymplectic

    GeometricIntegratorsBase.default_options(::TrialNN_PDE_int) = (
        max_iterations = 1,
        regularization_factor = 1e-5,
        verbosity = 0,
    )

    function run_case(; use_top_lifting::Bool, S::Int = 70, nsteps::Int = 2)
        x_span = (0.0, 1.0)
        h = 0.2

        lpde = MultiSymplectic.Wave.lpdeproblem(
            timestep = h,
            timespan = (0.0, nsteps * h),
            xspan = x_span,
            xstep = 0.01,
        )

        basis = Trial_Solution_Basis(S, tanh, x_span)
        method = TrialNN_PDE_int(
            basis;
            show_status = false,
            t_num_interval = 2,
            x_num_interval = 10,
            RT_per_interval = 4,
            RX_per_interval = 4,
            nx = 40,
            nt = 20,
            Nw = 500,
            Nb = 500,
            xspan = x_span,
        )

        int = MultiSymplectic.PDEIntegrator(lpde, method)
        sol = GeometricSolution(lpde)
        solstep = GeometricIntegratorsBase.solutionstep(int, sol[0])

        for _ in 1:nsteps
            if use_top_lifting
                TrialNNTopLiftingExperiment.integrate_top_lifting!(solstep, int)
            else
                MultiSymplectic.integrate!(solstep, int)
            end
        end

        xs = MultiSymplectic.cache(int).x_nodes
        u_err = maximum(abs.(current(solstep).u .- lpde.exact_u.(solstep.t, xs)))

        return (
            S = S,
            nsteps = nsteps,
            t = solstep.t,
            max_u_err = u_err,
            first_u = current(solstep).u[1],
            last_u = current(solstep).u[end],
        )
    end

    S = 70
    nsteps = 2
    old = run_case(use_top_lifting = false, S = S, nsteps = nsteps)
    new = run_case(use_top_lifting = true, S = S, nsteps = nsteps)

    println("old = ", old)
    println("new = ", new)
    println("error ratio old/new = ", old.max_u_err / new.max_u_err)
end
