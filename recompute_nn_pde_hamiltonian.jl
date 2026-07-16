using JLD2
using MultiSymplectic
using Logging
using Printf

const ERR_PATTERN = r"_err[^_]+_"
const NNINT_FILE_REGEX = r"^NNInt_fully_T(?<T>[^_]+)_h(?<h>[^_]+)_reg(?<reg>[^_]+)_S(?<S>[^_]+)_err(?<err>[^_]+)_Nw(?<Nw>\d+)_Nb(?<Nb>\d+)_tint(?<tint>\d+)_xint(?<xint>\d+)\.jld2$"

function format_err(err)
    return @sprintf("%.3e", err)
end

function parse_nnint_filename(path::AbstractString)
    name = basename(path)
    m = match(NNINT_FILE_REGEX, name)
    if m === nothing
        error("Filename does not match expected pattern: $name")
    end
    return (
        T = parse(Float64, m["T"]),
        h = parse(Float64, m["h"]),
        reg = parse(Float64, m["reg"]),
        S = parse(Int, m["S"]),
        err = parse(Float64, m["err"]),
        Nw = parse(Int, m["Nw"]),
        Nb = parse(Int, m["Nb"]),
        tint = parse(Int, m["tint"]),
        xint = parse(Int, m["xint"]),
    )
end

function construct_scaled_grid(t_num_interval::Int, x_num_interval::Int; xspan=(0.0, 1.0))
    grid_matrix, grid_weights = MultiSymplectic.construct_quadrature_grid([4, 4], [t_num_interval, x_num_interval])
    if ndims(grid_matrix) == 1
        RT = length(composite_quadrature(t_num_interval, 4).nodes)
        RX = length(composite_quadrature(x_num_interval, 4).nodes)
        grid_matrix = reshape(grid_matrix, RT, RX)
    end
    grid_matrix = [collect(grid_matrix[i, j]) for i in axes(grid_matrix, 1), j in axes(grid_matrix, 2)]
    x0 = xspan[1]
    x_domain = xspan[2] - xspan[1]
    @inbounds for k in eachindex(grid_matrix)
        t, xhat = grid_matrix[k]
        grid_matrix[k][2] = x0 + x_domain * xhat
    end
    grid_weights = reshape(grid_weights, size(grid_matrix))
    return grid_matrix, grid_weights
end

function hamiltonian_density(t, x, u, v, w, params)
    c= 4.0
    1 / 2 * (c * v[1]^2 + w[1]^2) - (1 + cos(u[1]))
end

function hamiltonian(u_quad_values::AbstractMatrix{<:Real}, v_quad_values::AbstractMatrix{<:Real}, w_quad_values::AbstractMatrix{<:Real},
    grid_quad_node::Matrix{Vector{Float64}}, grid_quad_weights::Matrix{Float64}, xspan::Tuple{Float64,Float64}, timestep::Float64, params)
    ham = 0.0
    RT, RX = size(u_quad_values)
    x_domain = xspan[2] - xspan[1]
    @inbounds for rt in 1:RT
        for rx in 1:RX
            ham += x_domain * timestep * grid_quad_weights[rt, rx] *
                hamiltonian_density(grid_quad_node[rt, rx][1], grid_quad_node[rt, rx][2],
                    u_quad_values[rt, rx], v_quad_values[rt, rx], w_quad_values[rt, rx], params)
        end
    end
    return ham
end

function recompute_for_file(path::AbstractString; apply::Bool = false)
    params = parse_nnint_filename(path)
    xspan = (0.0, 1.0)
    timespan = (0.0, params.T)
    lpde = MultiSymplectic.SineGordon.lpdeproblem(timestep = params.h, timespan = timespan, xspan = xspan)

    grid_matrix, grid_weights = construct_scaled_grid(params.tint, params.xint; xspan = xspan)

    data = with_logger(NullLogger()) do
        load(path)
    end

    nsteps = length(data["ham_ls"])
    times = collect(0.0:params.h:params.T - params.h)
    @assert length(times) == nsteps "Expected time steps length $(nsteps), got $(length(times))"

    ham_ls = zeros(Float64, nsteps)
    analytic_ham = zeros(Float64, nsteps)

    for (i, t0) in enumerate(times)
        u_quad = data["sol_u_quad"][i][1, :, :]
        v_quad = data["sol_v_quad"][i][1, :, :]
        w_quad = data["sol_w_quad"][i][1, :, :]

        analytic_u = Array{Float64}(undef, size(u_quad))
        analytic_v = Array{Float64}(undef, size(v_quad))
        analytic_w = Array{Float64}(undef, size(w_quad))
        RT, RX = size(u_quad)
        @inbounds for rt in 1:RT
            for rx in 1:RX
                local_t = t0 + params.h * grid_matrix[rt, rx][1]
                local_x = grid_matrix[rt, rx][2]
                analytic_u[rt, rx] = lpde.exact_u(local_t, local_x; params = lpde.params)
                analytic_v[rt, rx] = lpde.exact_v(local_t, local_x; params = lpde.params)
                analytic_w[rt, rx] = lpde.exact_w(local_t, local_x; params = lpde.params)
            end
        end

        analytic_ham[i] = hamiltonian(analytic_u, analytic_v, analytic_w, grid_matrix, grid_weights, xspan, params.h, lpde.params)
        ham_ls[i] = hamiltonian(u_quad, v_quad, w_quad, grid_matrix, grid_weights, xspan, params.h, lpde.params)
    end

    relative_ham_err = abs.((ham_ls .- analytic_ham) ./ analytic_ham)
    max_err = maximum(relative_ham_err)

    new_basename = replace(basename(path), ERR_PATTERN => "_err$(format_err(max_err))_")
    new_path = joinpath(dirname(path), new_basename)

    println("$(basename(path)) -> $(basename(new_path))")
    println("  recomputed max_err = $(format_err(max_err))")

    if apply
        jldopen(new_path, "w") do f
            for (k, v) in data
                if k == "ham_ls" || k == "analytic_ham" || k == "relative_ham_err" || k == "max_err"
                    continue
                end
                f[k] = v
            end
            f["ham_ls"] = ham_ls
            f["analytic_ham"] = analytic_ham
            f["relative_ham_err"] = relative_ham_err
            f["max_err"] = max_err
        end
        # rm(path)
    end

    return new_path, max_err, ham_ls, analytic_ham, relative_ham_err
end

function main()
    apply = "--apply" in ARGS
    paths = filter(x -> x != "--apply", ARGS)
    if isempty(paths)
        error("Usage: julia --project=. recompute_nn_pde_hamiltonian.jl [--apply] path1.jld2 [path2.jld2 ...]")
    end
    for path in paths
        recompute_for_file(path; apply = apply)
    end
    if !apply
        println("Dry run complete. Add --apply to write corrected files and rename originals.")
    end
end

main()
