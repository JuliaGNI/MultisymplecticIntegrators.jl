"""
    construct_quadrature_grid(R_list::Vector{Int},Num_intervals::Vector{Int})
Generate a grid of quadrature nodes and weights for multiple dimensions using composite Gauss-Legendre quadrature.
- `R_list`: vector specifying the number of quadrature points per interval for each dimension
- `Num_intervals`: vector specifying the number of intervals for each dimension
"""
function construct_quadrature_grid(R_list::Vector{Int}, Num_intervals_list::Vector{Int})
    @assert length(Num_intervals_list) == length(R_list) "Number of intervals must the length of R_list, should be the same as the dimension of the problem"
    quadrature_rules = [composite_quadrature(N, R)
                        for (N, R) in zip(Num_intervals_list, R_list)]

    # Extract nodes and weights for each dimension
    nodes = [rule.nodes for rule in quadrature_rules]
    weights = [rule.weights for rule in quadrature_rules]

    # Construct the grid of quadrature nodes using IterTools.product
    grid = collect(product(nodes...))

    # Compute the quadrature weights for the grid
    grid_weights = [prod(ws) for ws in product(weights...)]

    return grid, grid_weights
end

"""
    composite_quadrature(num_intervals, k::Int)

Composite Gauss–Legendre rule on [0, 1], split into `num_intervals` equal intervals with `k`
nodes in each. The rule on one interval is QuadratureRules' `GaussLegendreQuadrature(k; fast = true)`.
return: `(nodes, weights)`
"""
function composite_quadrature(num_intervals, k::Int)
    QGau = QuadratureRules.GaussLegendreQuadrature(k; fast = true)
    nodes_01, weights_01 = QGau.nodes, QGau.weights

    breaks = 0:(1 / num_intervals):1
    num_points_per_interval = length(nodes_01)

    total_points = num_intervals * num_points_per_interval
    all_nodes = zeros(total_points)
    all_weights = zeros(total_points)

    #scale and shift to each interval
    for i in 1:num_intervals
        x_left = breaks[i]
        x_right = breaks[i + 1]
        h = x_right - x_left

        start_idx = (i - 1) * num_points_per_interval + 1
        end_idx = i * num_points_per_interval

        # map the nodes ξ and weights ω of [0, 1] to x_left + h ξ and h ω
        all_nodes[start_idx:end_idx] .= x_left .+ h .* nodes_01
        all_weights[start_idx:end_idx] .= h .* weights_01
    end

    return (nodes = all_nodes, weights = all_weights)
end

function symbolize(p::Union{AbstractArray, Tuple}, name)
    vars = @variables $(name)[axes(p)...]
    first(vars)
end

parameter(name::Symbol) = Num(Symbolics.Sym{Real}(name))

function symbolize(::Number, name)
    parameter(name)
end

function symbolize(p::Union{Symbolics.Num, Symbolics.Arr{Symbolics.Num}}, name)
    p
end

function symbolize(params::NamedTuple)
    NamedTuple{keys(params)}(Tuple(symbolize(v, Symbol("$(k)ₚ"))
    for (k, v) in pairs(params)))
end

function substitute_parameters(code, params)
    if length(params) > 0
        # generate string of parameter arguments as they appear in generated code
        paramstr = string(Tuple(string(k) * "ₚ, " for k in keys(params))...)
        # remove trailing comma
        paramstr = paramstr[begin:prevind(paramstr, first((findlast(",", paramstr))))]
        # convert code expression to string
        code_str = string(code)
        # extract function header
        func_str = code_str[first(findfirst("function (", code_str)):last(findfirst(paramstr * ")", code_str))]
        # replace parameter list in function header with `params`
        func_str_params = replace(func_str, paramstr * ")" => "params)")
        # replace function header in code
        code_str = replace(code_str, func_str => func_str_params)
        # replace all params with named tuple entries
        for k in keys(params)
            code_str = replace(code_str, "$(k)ₚ" => "params.$(k)")
        end
        # convert code string back to expression
        return Meta.parse(code_str)
    else
        # convert code expression to string
        code_str = string(code)
        # extract function header
        func_str = code_str[first(findfirst("function (", code_str)):last(findfirst(")", code_str))]
        # append params argument to function header
        func_str_params = replace(func_str, ")" => ", params)")
        # replace function header in code
        code_str = replace(code_str, func_str => func_str_params)
        # convert code string back to expression
        return Meta.parse(code_str)
    end
end

function LPDE_variables(variable_dimension::Integer, x_domain_dimension::Integer)
    @variables t
    @variables x[1:x_domain_dimension]
    @variables U[1:variable_dimension]
    @variables V[1:variable_dimension]
    @variables W[1:variable_dimension]

    return (t, x, U, V, W)
end

"""
Basis for Lagrangian multipliers at the boundary, for 1D.
"""
function Lagrangian_multiplier(basis::Symbol, num_interval::Int, order::Integer, a, b)
    if basis ∉ (:BSplineDirichlet, :Lagrange)
        error("Unsupported basis: $basis")
    end
    Lagrangian_multiplier(Val(basis), num_interval, order, a, b)
end

function Lagrangian_multiplier(::Val{:BSplineDirichlet}, num_interval::Integer, k, a, b)
    BSplineDirichlet(num_interval, k, a, b)
end

struct BSplineDirichlet{TT, BT}
    k::Int # order

    N_intervals::Int
    t::TT # knots from BSplineKit
    b::BT #  basis functions

    Nbasis::Int # number of basis
    function BSplineDirichlet(num_interval::Int, k::Int, a, b)
        knot = a:((b - a) / (num_interval - 1)):b
        B = BSplineBasis(BSplineOrder(k), knot)

        return new{typeof(B.t), typeof(B)}(k, num_interval, B.t, B, length(B))
    end
end

Base.length(Basis::BSplineDirichlet) = Basis.Nbasis

function Base.show(io::IO, basis::BSplineDirichlet)
    print(io, "\n Dirichlet B-spline Multiplier Basis with:\n")
    print(io, "   Order k: $(basis.k) \n")
    print(io, "   Number of intervals: $(basis.N_intervals) \n")
    print(io, "   Number of basis functions: $(basis.Nbasis) \n")
    print(io, "   Knots: $(basis.t) \n")
end

function Lagrangian_multiplier(::Val{:Lagrange}, num_interval::Int, k::Integer, a, b)
    QGau = QuadratureRules.GaussLegendreQuadrature(num_interval+1)
    CompactBasisFunctions.Lagrange(a .+ (b - a) .* QGau.nodes)
end

function initialize_bcs_ics!(sol, int::PDEIntegrator)
    local C = cache(int)
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local D = int.problem.D
    local RT = int.method.RT
    local ic_fun = int.problem.ics_function
    local tn = sol.t - timestep(int)
    local bc_fun = int.problem.bcs_function
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local has_λ₀_coes = hasproperty(C, :λ₀_x_coes) && hasproperty(int.method, :mλ_x)
    local mλ_x = has_λ₀_coes ? int.method.mλ_x : nothing
    local lag_sys = has_λ₀_coes ? int.problem.lagrangian_system.functions : nothing
    local lag_params = has_λ₀_coes ? int.problem.lagrangian_system.params : nothing
    local carried_internal = internal(sol)
    for d in 1:D
        if iszero(tn)
            ic_vals = ic_fun(xspan[1] .+ x_domain .* x_quad_nodes)
            C.init_condition_t₀[d, :] .= ic_vals.u

            C.ics_ut₀_quad_values[d, :] .= ic_vals.u
            C.ics_vt₀_quad_values[d, :] .= ic_vals.v
            C.ics_wt₀_quad_values[d, :] .= ic_vals.w

            if has_λ₀_coes
                momentum = Vector{eltype(C.λ₀_x_coes)}(undef, size(mλ_x, 2))
                for rx in eachindex(momentum)
                    momentum[rx] = lag_sys.∂L∂V[d](
                        C.ics_ut₀_quad_values[d, rx], C.ics_vt₀_quad_values[d, rx],
                        C.ics_wt₀_quad_values[d, rx], lag_params)
                end
                @views C.λ₀_x_coes[d, :] .= mλ_x' \ momentum
            end
        else
            C.init_condition_t₀[d, :] = carried_internal.ut₁_quad_values[d, :]

            C.ics_ut₀_quad_values[d, :] = carried_internal.ut₁_quad_values[d, :]
            C.ics_vt₀_quad_values[d, :] = carried_internal.vt₁_quad_values[d, :]
            C.ics_wt₀_quad_values[d, :] = carried_internal.wt₁_quad_values[d, :]

            if has_λ₀_coes && haskey(carried_internal, :λ₁_x_coes)
                @views C.λ₀_x_coes[d, :] .= carried_internal.λ₁_x_coes[d, :]
            end
        end

        for i in 1:RT
            bc = bc_fun(tn + timestep(int) * t_quad_nodes[i], xspan)
            C.boundary_condition_x₀[d, i] = bc.bc₀.u
            C.boundary_condition_x₁[d, i] = bc.bc₁.u

            C.bc_ux₀_quad_values[d, i] = bc.bc₀.u
            C.bc_vx₀_quad_values[d, i] = bc.bc₀.v
            C.bc_wx₀_quad_values[d, i] = bc.bc₀.w

            C.bc_ux₁_quad_values[d, i] = bc.bc₁.u
            C.bc_vx₁_quad_values[d, i] = bc.bc₁.v
            C.bc_wx₁_quad_values[d, i] = bc.bc₁.w
        end
    end
end

function internal_variables(method::PDEMethod, problem::LPDEProblem)
    local D = problem.D
    local RX = method.RX

    ut₁_quad_values = zeros(D, RX)
    vt₁_quad_values = zeros(D, RX)
    wt₁_quad_values = zeros(D, RX)

    return (ut₁_quad_values = ut₁_quad_values,
        vt₁_quad_values = vt₁_quad_values,
        wt₁_quad_values = wt₁_quad_values
    )
end

function copy_internal_variables!(solstep::SolutionStep, C::PDEIntegratorCache)
    # copy internal variables from cache to internal,
    haskey(internal(solstep), :ut₁_quad_values) &&
        copyto!(internal(solstep).ut₁_quad_values, C.ut₁_quad_values)
    haskey(internal(solstep), :vt₁_quad_values) &&
        copyto!(internal(solstep).vt₁_quad_values, C.vt₁_quad_values)
    haskey(internal(solstep), :wt₁_quad_values) &&
        copyto!(internal(solstep).wt₁_quad_values, C.wt₁_quad_values)
    haskey(internal(solstep), :λ₁_x_coes) && hasproperty(C, :λ₁_x_coes) &&
        copyto!(internal(solstep).λ₁_x_coes, C.λ₁_x_coes)
end

function flatten_params(params::NeuralNetworkParameters)
    flat_list = []
    for layer in values(params)
        for field in fieldnames(typeof(layer))
            val = getfield(layer, field)
            push!(flat_list, vec(val))
        end
    end
    return vcat(flat_list...)
end

function reconstruct_params(flat, template::NeuralNetworkParameters)
    idx = 1
    reconstructed = NamedTuple()

    # Iterate over the outer NamedTuple (layers)
    layers = map(values(template)) do layer
        # Iterate over the inner NamedTuple (fields like weights, biases)
        fields = map(fieldnames(typeof(layer))) do fname
            original_val = getfield(layer, fname)
            len = length(original_val)
            reconstructed_val = reshape(flat[idx:(idx + len - 1)], size(original_val))
            idx += len
            fname => reconstructed_val
        end
        NamedTuple(fields)
    end

    # Reconstruct using the original keys of the outer NamedTuple
    named_keys = keys(template)
    return NamedTuple{named_keys}(layers)
end

using IterTools

function construct_quadrature_grid_with_boundary(dimensions::Vector{Int})
    #including the quadrature points on the boundary and in the interior.
    d = length(dimensions)

    # Create quadrature rules for each dimension
    quad_rules(R) = QuadratureRules.GaussLegendreQuadrature(R; fast = true)

    quadrature_rules = [quad_rules(R) for R in dimensions]
    nodes = [rule.nodes for rule in quadrature_rules]
    weights = [rule.weights for rule in quadrature_rules]

    # Interior grid
    interior_grid = collect(product(nodes...))
    interior_weights = [prod(w) for w in product(weights...)]

    full_grid = Vector{Vector{Float64}}()
    full_weights = Float64[]

    # Add interior points
    for (pt, w) in zip(interior_grid, interior_weights)
        push!(full_grid, collect(pt))
        push!(full_weights, w)
    end

    # Add boundary faces
    for i in 1:d
        other_indices = setdiff(1:d, [i])
        nodes_rest = nodes[other_indices]
        weights_rest = weights[other_indices]

        subgrid = collect(product(nodes_rest...))
        subweights = [prod(w) for w in product(weights_rest...)]

        for fixed_val in (0.0, 1.0)
            for (pt, w) in zip(subgrid, subweights)
                new_point = Vector{Float64}(undef, d)
                k = 1  # index for accessing elements of pt
                for j in 1:d
                    if j == i
                        new_point[j] = fixed_val
                    else
                        new_point[j] = pt[k]
                        k += 1
                    end
                end
                push!(full_grid, new_point)
                push!(full_weights, w)
            end
        end
    end

    # Convert to matrix where each column is a point
    grid_matrix = hcat(full_grid...)

    @assert size(grid_matrix, 2) == length(full_weights) "Mismatch between number of grid points and weights."

    return grid_matrix, full_weights
end

function lsgd_loss(network_inputs, labels, NN, ps)
    NN_output = NN(network_inputs, ps)
    return Statistics.mean((labels .- NN_output) .^ 2)
end

function create_interior_quadrature_points_derivative_mat(
        ST::Type, RT::Int, RX::Int, D::Int, P_sizes::Vector{Int})
    Ptotal = sum(P_sizes)

    mat = zeros(ST, RT, RX, Ptotal, D)

    offset = vcat(1, cumsum(P_sizes) .+ 1)

    mat_view = Vector{AbstractArray{ST, 3}}(undef, D)

    for d in 1:D
        mat_view[d] = @view mat[:, :, offset[d]:(offset[d + 1] - 1), d]
    end

    return mat_view
end

function create_interior_quadrature_points_derivative_mat(
        ST::Type, RT::Int, RX::Int, D::Int, DX::Int, P_sizes::Vector{Int})
    Ptotal = sum(P_sizes)

    mat = zeros(ST, RT, RX, Ptotal, D, DX)

    offset = vcat(1, cumsum(P_sizes) .+ 1)

    mat_view = Matrix{AbstractArray{ST, 3}}(undef, D, DX)

    for d in 1:D
        for dx in 1:DX
            mat_view[d, dx] = @view mat[:, :, offset[d]:(offset[d + 1] - 1), d, dx]
        end
    end

    return mat_view
end

function create_boundary_derivative_vector(ST::Type, D::Int, R::Int, P_sizes::Vector{Int})
    Ptotal = sum(P_sizes)
    mat = zeros(ST, R, Ptotal, D)

    offset = vcat(1, cumsum(P_sizes) .+ 1)

    mat_view = Vector{Any}(undef, D)

    for d in 1:D
        mat_view[d] = @view mat[:, offset[d]:(offset[d + 1] - 1), d]
    end

    return mat_view
end

function eval_spline2D(coefs::AbstractMatrix, (Bt, Bx), (t, x))
    it, bt = Bt(t)
    ix, bx = Bx(x)

    kt = BSplineKit.order(Bt)
    kx = BSplineKit.order(Bx)
    val = zero(eltype(coefs))  # spline evaluated at (t, x)
    Nt, Nx = length(Bt), length(Bx)
    for δx in eachindex(bx), δt in eachindex(bt)

        ii = it - δt + 1
        jj = ix - δx + 1
        if ii >= 1 && ii <= Nt && jj >= 1 && jj <= Nx
            coef = coefs[it - δt + 1, ix - δx + 1]
            val += coef * bt[δt] * bx[δx]
        end
    end
    val
end

function eval_spline2D_dt(coefs::AbstractMatrix, (Bt, Bx), (t, x))
    it, bt = Bt(t)
    it_d, btd = Bt(t, BSplineKit.Derivative(1))  # N'(t)
    @assert it == it_d
    ix, bx = Bx(x)

    kt = BSplineKit.order(Bt)
    kx = BSplineKit.order(Bx)
    val = zero(eltype(coefs))  # spline evaluated at (t, x)
    Nt, Nx = length(Bt), length(Bx)
    for δx in eachindex(bx), δt in eachindex(btd)

        ii = it_d - δt + 1
        jj = ix - δx + 1
        if ii >= 1 && ii <= Nt && jj >= 1 && jj <= Nx
            coef = coefs[it_d - δt + 1, ix - δx + 1]
            val += coef * btd[δt] * bx[δx]
        end
    end
    val
end

function eval_spline2D_dx(coefs::AbstractMatrix, (Bt, Bx), (t, x))
    it, bt = Bt(t)

    ix, bx = Bx(x)
    ix_d, bxd = Bx(x, BSplineKit.Derivative(1))   # M'(x)

    @assert ix == ix_d

    kt = BSplineKit.order(Bt)
    kx = BSplineKit.order(Bx)
    val = zero(eltype(coefs))  # spline evaluated at (t, x)
    Nt, Nx = length(Bt), length(Bx)
    for δx in eachindex(bxd), δt in eachindex(bt)

        ii = it - δt + 1
        jj = ix_d - δx + 1
        if ii >= 1 && ii <= Nt && jj >= 1 && jj <= Nx
            coef = coefs[ii, jj]
            val += coef * bt[δt] * bxd[δx]
        end
    end
    val
end

function create_tem_vector(ST::Type, D::Int, P_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, P_sizes[d]))
    end
    return mat
end

# ∂S/∂c
function spline2D_active_coeff_derivatives(
        (Bt, Bx)::Tuple{BTT, BXT}, (t, x)::Tuple{Float64, Float64}) where {BXT, BTT}
    it, bt = Bt(t)
    ix, bx = Bx(x)
    Nt, Nx = length(Bt), length(Bx)
    indices = Int[]
    values = Float64[]
    for δx in eachindex(bx), δt in eachindex(bt)

        ii = it - δt + 1
        jj = ix - δx + 1
        if ii >= 1 && ii <= Nt && jj >= 1 && jj <= Nx
            push!(indices, ii + (jj - 1) * Nt)
            push!(values, bt[δt] * bx[δx])
        end
    end
    return indices, values
end
function spline2D_active_coeff_derivatives((Bt, Bx)::Tuple{BTT, BXT}, tx::Vector{Float64}) where {
        BXT, BTT}
    spline2D_active_coeff_derivatives((Bt, Bx), (tx[1], tx[2]))
end

# ∂v/∂c = N'_i(t) * M_j(x)
function spline2D_active_coeff_derivatives_time(
        (Bt, Bx)::Tuple{BTT, BXT}, (t, x)::Tuple{Float64, Float64}) where {BXT, BTT}
    it_d, btd = Bt(t, BSplineKit.Derivative(1))
    ix, bx = Bx(x)
    Nt, Nx = length(Bt), length(Bx)
    indices = Int[]
    values = Float64[]
    for δx in eachindex(bx), δt in eachindex(btd)

        ii = it_d - δt + 1
        jj = ix - δx + 1
        if ii >= 1 && ii <= Nt && jj >= 1 && jj <= Nx
            push!(indices, ii + (jj - 1) * Nt)
            push!(values, btd[δt] * bx[δx])
        end
    end
    return indices, values
end
function spline2D_active_coeff_derivatives_time((Bt, Bx)::Tuple{BTT, BXT}, tx::Vector{Float64}) where {
        BXT, BTT}
    spline2D_active_coeff_derivatives_time((Bt, Bx), (tx[1], tx[2]))
end

# ∂w/∂c = N_i(t) * M'_j(x)
function spline2D_active_coeff_derivatives_space(
        (Bt, Bx)::Tuple{BTT, BXT}, (t, x)::Tuple{Float64, Float64}) where {BXT, BTT}
    it, bt = Bt(t)
    ix_d, bxd = Bx(x, BSplineKit.Derivative(1))
    Nt, Nx = length(Bt), length(Bx)
    indices = Int[]
    values = Float64[]
    for δx in eachindex(bxd), δt in eachindex(bt)

        ii = it - δt + 1
        jj = ix_d - δx + 1
        if ii >= 1 && ii <= Nt && jj >= 1 && jj <= Nx
            push!(indices, ii + (jj - 1) * Nt)
            push!(values, bt[δt] * bxd[δx])
        end
    end
    return indices, values
end
function spline2D_active_coeff_derivatives_space(
        (Bt, Bx)::Tuple{BTT, BXT}, tx::Vector{Float64}) where {BXT, BTT}
    spline2D_active_coeff_derivatives_space((Bt, Bx), (tx[1], tx[2]))
end

function vector_hessian(f, x)
    S = length(f(x))
    D = length(x)
    out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
    return reshape(out, S, D, D)  # return a 3D array where out[i, j, k] = ∂²f_i / ∂x_j ∂x_k
end
