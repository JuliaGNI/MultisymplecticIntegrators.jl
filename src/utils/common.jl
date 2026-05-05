
function GaussQuadrature64()
    points = [0.99930504173577213946, 0.99634011677195527935, 0.99101337147674432074, 0.98333625388462595693, 0.97332682778991096374, 0.96100879965205371892, 0.94641137485840281606, 0.92956917213193957582, 0.91052213707850280576, 0.88931544599511410585, 0.86599939815409281976, 0.84062929625258036275, 0.81326531512279755974, 0.78397235894334140761, 0.75281990726053189661, 0.71988185017161082685, 0.68523631305423324256, 0.64896547125465733986, 0.61115535517239325025, 0.57189564620263403428, 0.53127946401989454566, 0.48940314570705295748, 0.44636601725346408798, 0.4022701579639916037, 0.35722015833766811595, 0.31132287199021095616, 0.26468716220876741637, 0.21742364374000708415, 0.16964442042399281804, 0.12146281929612055447, 0.07299312178779903945, 0.024350292663424432509]
    weight = [0.0017832807216964329473, 0.0041470332605624676353, 0.0065044579689783628561, 0.0088467598263639477231, 0.011168139460131128819, 0.013463047896718642598, 0.015726030476024719322, 0.017951715775697343085, 0.020134823153530209372, 0.022270173808383254159, 0.024352702568710873338, 0.026377469715054658672, 0.028339672614259483228, 0.030234657072402478868, 0.032057928354851553585, 0.033805161837141609392, 0.035472213256882383811, 0.03705512854024004604, 0.038550153178615629129, 0.039953741132720341387, 0.04126256324262352861, 0.042473515123653589007, 0.043583724529323453377, 0.04459055816375656306, 0.04549162792741814448, 0.046284796581314417296, 0.046968182816210017325, 0.047540165714830308662, 0.047999388596458307728, 0.04834476223480295717, 0.048575467441503426935, 0.048690957009139720383]
    
    nodes = 0.5 .* ([-1 .* points...,reverse(points)...] .+ 1)
    weights = 0.5 .* [weight...,reverse(weight)...]
    return (nodes = nodes, weights = weights)
end
function GaussQuadrature128()
    points = [0.99982488794713191447, 0.99907745997737589501, 0.99773324862551401988, 0.99579275853498118687, 0.9932571129002129353, 0.99012781849173438334, 0.98640674272458620887, 0.98209610843571853603, 0.97719849146390738716, 0.9717168187471365809, 0.96565436643196526864, 0.9590147578536999281, 0.95180196134126438622, 0.94402028783022018212, 0.93567438827791637578, 0.92676925087894784333, 0.91731019808096053704, 0.90730288340175681392, 0.89675328804915818439, 0.88566771734539721741, 0.8740527969580317987, 0.86191546893954846059, 0.84926298757796896916, 0.83610291506090684712, 0.82244311695564384246, 0.80829175750791366012, 0.79365729476219329024, 0.77854847550641196685, 0.76297433004409472278, 0.74694416679706198117, 0.73046756674190880647, 0.71355437768358741334, 0.69621470836951433239, 0.67845892244771925937, 0.66029763227264605211, 0.64174169256230755715, 0.62280219391058491076, 0.6034904561585486242, 0.58381802162876308955, 0.56379664822661808391, 0.54343830241281036344, 0.52275515205117547845, 0.50175955913614446429, 0.48046407240417202586, 0.45888141983355219545, 0.43702450103710416294, 0.41490637955227501549, 0.39254027503326744274, 0.36993955534985902662, 0.34711772859763550843, 0.32408843502441337518, 0.30086543887767720267, 0.27746262017790440281, 0.25389396642269432086, 0.23017356422665998641, 0.20631559090207921715, 0.18233430598533718241, 0.158244042714224934, 0.13405919946118778512, 0.10979423112764374667, 0.085463640504515498637, 0.061081969604139568104, 0.03666379096873349333, 0.012223698960615764198]
    weight = [0.00044938096029209037639, 0.0010458126793403487793, 0.0016425030186690295388, 0.0022382884309626187436, 0.0028327514714579910953, 0.0034255260409102157743, 0.0040162549837386423132, 0.0046045842567029551183, 0.0051901618326763302051, 0.0057726375428656985893, 0.0063516631617071887872, 0.0069268925668988135634, 0.0074979819256347286877, 0.0080645898904860579729, 0.008626377798616749705, 0.0091830098716608743345, 0.0097341534150068058636, 0.010279479015832157133, 0.010818660739503076248, 0.011351376324080416693, 0.011877307372740279576, 0.012396139543950922969, 0.01290756273926734722, 0.013411271288616332315, 0.013906964132951985244, 0.014394345004166846177, 0.014873122602147314252, 0.015343010768865144086, 0.015803728659399346859, 0.016255000909785187052, 0.016696557801589204589, 0.017128135423111376831, 0.017549475827117704649, 0.01796032718500868594, 0.018360443937331343221, 0.018749586940544708651, 0.019127523609950945487, 0.019494028058706602823, 0.01984888123283086222, 0.020191871042130041181, 0.020522792486960069432, 0.020841447780751149114, 0.021147646468221348537, 0.021441205539208460137, 0.021721949538052075375, 0.021989710668460491434, 0.022244328893799765105, 0.022485652032744966872, 0.02271353585023646131, 0.02292784414368684692, 0.023128448824387027879, 0.023315229994062760122, 0.023488076016535913153, 0.023646883584447615144, 0.023791557781003400639, 0.023922012136703455672, 0.024038168681024052638, 0.024139957989019284998, 0.02422731922281524812, 0.024300200167971865323, 0.024358557264690625853, 0.024402355633849582093, 0.024431569097850045055, 0.024446180196262518211]
    
    nodes = 0.5 .* ([-1 .* points...,reverse(points)...] .+ 1)
    weights = 0.5 .* [weight...,reverse(weight)...]
    return (nodes = nodes, weights = weights)
end

"""
    construct_quadrature_grid(R_list::Vector{Int},Num_intervals::Vector{Int})
Generate a grid of quadrature nodes and weights for multiple dimensions using composite Gauss-Legendre quadrature.
- `R_list`: vector specifying the number of quadrature points per interval for each dimension
- `Num_intervals`: vector specifying the number of intervals for each dimension
"""
function construct_quadrature_grid(R_list::Vector{Int},Num_intervals_list::Vector{Int})

    @assert length(Num_intervals_list) == length(R_list) "Number of intervals must the length of R_list, should be the same as the dimension of the problem"
    quadrature_rules = [composite_quadrature(N,R) for (N,R) in zip(Num_intervals_list,R_list)]

    # Extract nodes and weights for each dimension
    nodes = [rule.nodes for rule in quadrature_rules]
    weights = [rule.weights for rule in quadrature_rules]

    # Construct the grid of quadrature nodes using IterTools.product
    grid = collect(product(nodes...))

    # Convert the grid into a matrix where each column is a grid point
    # grid_matrix = hcat(map(x -> collect(x), grid)...)

    # Compute the quadrature weights for the grid
    grid_weights = [prod(ws) for ws in product(weights...)]
    # grid_weights = hcat(map(x -> collect(x), grid_weights)...)

    return grid, grid_weights
end


"""
    composite_quadrature(breaks::Vector{T}, k::Int) where T<:Real
    
Generate composite Gauss-Legendre quadrature nodes and weights over multiple intervals defined by `breaks`.
- `breaks`: strictly increasing vector [x1, x2, ..., xn]
- `k`: order of QuadratureRules
return: (all_nodes, all_weights)
"""
function composite_quadrature(num_intervals, k::Int)
    if k == 128
        QGau = GaussQuadrature128()
    elseif k == 64
        QGau = GaussQuadrature64()
    else
        QGau = QuadratureRules.GaussLegendreQuadrature(k)
    end
    nodes_01, weights_01 = QGau.nodes, QGau.weights

    # num_intervals = length(breaks) - 1
    breaks = 0:1/num_intervals:1
    num_points_per_interval = length(nodes_01)
    
    total_points = num_intervals * num_points_per_interval
    all_nodes = zeros(total_points)
    all_weights = zeros(total_points)
    
    #scale and shift to each interval
    for i in 1:num_intervals
        x_left = breaks[i]
        x_right = breaks[i+1]
        h = x_right - x_left  
        
        start_idx = (i - 1) * num_points_per_interval + 1
        end_idx = i * num_points_per_interval
        
        # X = x_left + h * ξ
        # W = h * ω
        all_nodes[start_idx:end_idx] .= x_left .+ h .* nodes_01
        all_weights[start_idx:end_idx] .= h .* weights_01
    end
    
    return (nodes = all_nodes, weights = all_weights)
end

# # Example usage for 2 dimensions
# dimensions = [2,3,4]  # Number of quadrature points in each dimension
# grid_matrix, grid_weights = construct_quadrature_grid(dimensions)

# # Print results
# println("Grid of quadrature nodes:")
# println(grid_matrix)

# println("\nQuadrature weights for the grid:")
# println(grid_weights)


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
    NamedTuple{keys(params)}(Tuple(symbolize(v, Symbol("$(k)ₚ")) for (k,v) in pairs(params)))
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


function LPDE_variables(variable_dimension::Integer,x_domain_dimension::Integer)
    @variables t
    @variables x[1:x_domain_dimension]
    # @variables (u(x...,t))[1:variable_dimension]     #@variables (u(sym_x,sym_t))[1:variable_dimension] to not expand spatial variable x
    # @variables (v(x...,t))[1:variable_dimension]
    # @variables (w(x...,t))[1:variable_dimension,1:x_domain_dimension] # what is the dimension of w?
    @variables U[1:variable_dimension]     
    @variables V[1:variable_dimension]
    @variables W[1:variable_dimension]
    # @variables W[1:variable_dimension,1:x_domain_dimension]

    return (t, x, U, V, W)
end

function lagrangianPDE_derivatives(t,x,u,v,w)

    Dt = Differential(t)
    Dx = collect(Differential.(x))
    Du = collect(Differential.(u))
    Dv = collect(Differential.(v))
    Dw = collect(Differential.(w))
    # Dx = Differential(x)
    # Du = Differential(u)
    # Dv = Differential(v)
    # Dw = Differential(w)

    return (Dt, Dx, Du, Dv, Dw)
end


"""
Basis for Lagrangian multipliers at the boundary, for 1D.
"""
function Lagrangian_multiplier(basis::Symbol, num_interval::Int, order::Integer,a,b)
    if basis ∉ (:BSplineDirichlet, :Lagrange)
        error("Unsupported basis: $basis")
    end
    Lagrangian_multiplier(Val(basis), num_interval,order,a,b)
end

Lagrangian_multiplier(::Val{:BSplineDirichlet},num_interval::Integer,k,a,b) = BSplineDirichlet(num_interval,k,a,b)

struct BSplineDirichlet 
    k::Int # order

    N_intervals::Int
    t::AbstractVector # knots from BSplineKit
    b #  basis functions 

    Nbasis::Int # number of basis
    function BSplineDirichlet(num_interval::Int,k::Int,a,b)
        knot = a:(b - a)/(num_interval-1):b
        B = BSplineBasis(BSplineOrder(k), knot)
    
        return new(k, num_interval,B.t, B, length(B))
    end
end

Base.length(Basis::BSplineDirichlet) = Basis.Nbasis

function Lagrangian_multiplier(::Val{:Lagrange},num_interval::Int,k::Integer,a,b) 
    QGau = QuadratureRules.GaussLegendreQuadrature(num_interval+1)
    CompactBasisFunctions.Lagrange(a .+ (b - a) .* QGau.nodes)
end


function initialize_bcs_ics!(sol,int::PDEIntegrator)
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
    for d in 1:D
        # println("update initial condition, current time = ", sol.t, "the initial condition is at time = ", sol.t - timestep(int))

        if tn == 0.0
            C.init_condition_t₀[d,:] .= ic_fun(xspan[1] .+ x_domain .* x_quad_nodes).u

            C.ics_ut₀_quad_values[d,:] .= ic_fun(xspan[1] .+ x_domain .* x_quad_nodes).u
            C.ics_vt₀_quad_values[d,:] .= ic_fun(xspan[1] .+ x_domain .* x_quad_nodes).v
            C.ics_wt₀_quad_values[d,:] .= ic_fun(xspan[1] .+ x_domain .* x_quad_nodes).w
        else
            C.init_condition_t₀[d,:] = internal(sol).ut₁_quad_values[d,:]

            C.ics_ut₀_quad_values[d,:] = internal(sol).ut₁_quad_values[d,:]
            C.ics_vt₀_quad_values[d,:] = internal(sol).vt₁_quad_values[d,:]
            C.ics_wt₀_quad_values[d,:] = internal(sol).wt₁_quad_values[d,:]
        end

        for i in 1:RT
            C.boundary_condition_x₀[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₀.u
            C.boundary_condition_x₁[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₁.u

            C.bc_ux₀_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₀.u
            C.bc_vx₀_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₀.v
            C.bc_wx₀_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₀.w

            C.bc_ux₁_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₁.u
            C.bc_vx₁_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₁.v
            C.bc_wx₁_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₁.w
        end
        # println("left boundary condition = " , C.boundary_condition_x₀[d,:])
    end

end

function internal_variables(method::PDEMethod, problem::LPDEProblem)
    local D = problem.D
    local RX = method.RX

    ut₁_quad_values = zeros(D,RX)
    vt₁_quad_values = zeros(D,RX)
    wt₁_quad_values = zeros(D,RX)
    
    return (ut₁_quad_values = ut₁_quad_values,
        vt₁_quad_values = vt₁_quad_values,
        wt₁_quad_values = wt₁_quad_values,
        )
end

function copy_internal_variables!(solstep::SolutionStep,C::PDEIntegratorCache)
    # copy internal variables from cache to internal,
    haskey(internal(solstep), :ut₁_quad_values) && copyto!(internal(solstep).ut₁_quad_values,C.ut₁_quad_values)
    haskey(internal(solstep), :vt₁_quad_values) && copyto!(internal(solstep).vt₁_quad_values,C.vt₁_quad_values)
    haskey(internal(solstep), :wt₁_quad_values) && copyto!(internal(solstep).wt₁_quad_values,C.wt₁_quad_values)
end

# flat = flatten_params(pnn.params)
# reconstructed = reconstruct_params(flat, pnn.params)

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
            reconstructed_val = reshape(flat[idx:idx+len-1], size(original_val))
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
    function quad_rules(R)
        if R == 128
            return GaussQuadrature128()
        elseif R == 64
            return GaussQuadrature64()
        else
            return QuadratureRules.GaussLegendreQuadrature(R)
        end
    end

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


function box_init_plain(input_dim::Int, output_dim::Int;Random_rng = Random.seed!(1))
    W = zeros(Float32, output_dim, input_dim)
    b = zeros(Float32, output_dim)

    for i in 1:output_dim
        p = rand(Random_rng,Float32, input_dim) 
        n = randn(Random_rng,Float32, input_dim)
        n ./= norm(n)
        p_max = map((n_i) -> n_i ≥ 0 ? 1.0f0 : 0.0f0, n)
        k = 1 / dot((p_max .- p), n)
        W[i, :] = k * n
        b[i] = k * dot(p, n)
    end
    return W, b
    # initialize the parameters and train with LSGD
    # for (name, layer) in zip(keys(PNN.params), values(PNN.params))
    #     in_size = size(layer.W, 2)
    #     out_size = size(layer.W, 1)
    #     if hasfield(typeof(layer), :b)
    #         layer.W[:], layer.b[:] = box_init_plain(in_size, out_size)
    #     else
    #         # For layers without bias (e.g., output), just regenerate W
    #         layer.W[:], _ = box_init_plain(in_size, out_size)
    #     end
    # end

end

function lsgd_loss(network_inputs,labels,NN,ps)
    NN_output = NN(network_inputs, ps)
    return Statistics.mean((labels .- NN_output).^2)
end

function create_interior_quadrature_points_derivative_mat(ST::Type, RT::Int, RX::Int, D::Int, P_sizes::Vector{Int})

    Ptotal = sum(P_sizes)

    mat = zeros(ST, RT, RX, Ptotal, D)

    offset = vcat(1, cumsum(P_sizes) .+ 1)

    mat_view = Vector{AbstractArray{ST,3}}(undef, D)

    for d in 1:D
        mat_view[d] = @view mat[:, :, offset[d]:offset[d+1]-1, d]
    end

    return mat_view
end

function create_interior_quadrature_points_derivative_mat(ST::Type, RT::Int, RX::Int, D::Int, DX::Int, P_sizes::Vector{Int})
    Ptotal = sum(P_sizes)

    mat = zeros(ST, RT, RX, Ptotal, D, DX)

    offset = vcat(1, cumsum(P_sizes) .+ 1)

    mat_view = Matrix{AbstractArray{ST,3}}(undef, D, DX)

    for d in 1:D
        for dx in 1:DX
            mat_view[d, dx] =
                @view mat[:, :, :, offset[d]:offset[d+1]-1, d, dx]
        end
    end

    return mat_view
end

function create_boundary_derivative_vector(ST::Type, D::Int,R::Int,P_sizes::Vector{Int})
    Ptotal = sum(P_sizes)
    mat = zeros(ST, R, Ptotal, D)

    offset = vcat(1, cumsum(P_sizes) .+ 1)

    mat_view = Vector{Any}(undef, D)

    for d in 1:D
        mat_view[d] = @view mat[:, offset[d]:offset[d+1]-1, d]
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

    ix, bx = Bx(x)

    kt = BSplineKit.order(Bt)
    kx = BSplineKit.order(Bx)
    val = zero(eltype(coefs))  # spline evaluated at (t, x)
    Nt, Nx = length(Bt), length(Bx)
    for δx in eachindex(bx), δt in eachindex(btd)
        ii = it - δt + 1
        jj = ix - δx + 1
        if ii >= 1 && ii <= Nt && jj >= 1 && jj <= Nx
            coef = coefs[it - δt + 1, ix - δx + 1]
            val += coef * btd[δt] * bx[δx]
        end
    end
    val
end

function eval_spline2D_dx(coefs::AbstractMatrix, (Bt, Bx), (t, x))
    it, bt = Bt(t)

    ix, bx = Bx(x)
    ix_d, bxd = Bx(x, BSplineKit.Derivative(1))   # M'(x)

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

function create_tem_vector(ST::Type, D::Int,P_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, P_sizes[d]))
    end
    return mat
end


function spline2D_all_derivatives((Bt, Bx)::Tuple{BTT,BXT}, (t, x)::Tuple{Float64,Float64}) where {BXT,BTT}
    # basis in t
    it, bt = Bt(t)
    it_d, btd = Bt(t, BSplineKit.Derivative(1))   # N'(t)

    # basis in x
    ix, bx = Bx(x)
    ix_d, bxd = Bx(x, BSplineKit.Derivative(1))   # M'(x)

    kt = order(Bt)
    kx = order(Bx)

    # allocate outputs
    dSdc = zeros(length(Bt), length(Bx))
    dVdc = similar(dSdc)
    dWdc = similar(dSdc)

    @inbounds for δx in 1:kx, δt in 1:kt
        ii = it - δt + 1
        jj = ix - δx + 1

        # u(t,x)
        dSdc[ii, jj] = bt[δt] * bx[δx]

        # ∂u/∂t(t,x) 
        dVdc[ii, jj] = btd[δt] * bx[δx]

        # ∂u/∂x(t,x)
        dWdc[ii, jj] = bt[δt] * bxd[δx]
    end

    return reshape(dSdc, :, ), reshape(dVdc, :, ), reshape(dWdc, :, )
end
spline2D_all_derivatives((Bt, Bx)::Tuple{BTT,BXT}, tx::Vector{Float64}) where {BXT,BTT} = spline2D_all_derivatives((Bt, Bx), (tx[1], tx[2]))
# ∂S/∂c
function spline2D_coeff_derivatives((Bt, Bx)::Tuple{BTT,BXT}, (t, x)::Tuple{Float64,Float64}) where {BXT,BTT}
    it, bt = Bt(t)
    ix, bx = Bx(x)

    kt = BSplineKit.order(Bt)
    kx = BSplineKit.order(Bx)

    Nt, Nx = length(Bt), length(Bx)
    dSdc = zeros(Nt, Nx)
    for δx in eachindex(bx), δt in eachindex(bt)
            ii = it - δt + 1
            jj = ix - δx + 1
            if ii >= 1 && ii <= Nt && jj >= 1 && jj <= Nx
            dSdc[ii, jj] = bt[δt] * bx[δx]
        end
    end
    return reshape(dSdc, :, )
end
spline2D_coeff_derivatives((Bt, Bx)::Tuple{BTT,BXT}, tx::Vector{Float64}) where {BXT,BTT} = spline2D_coeff_derivatives((Bt, Bx), (tx[1], tx[2]))


# ∂v/∂c = N'_i(t) * M_j(x) 
function spline2D_coeff_derivatives_time((Bt, Bx)::Tuple{BTT,BXT}, (t, x)::Tuple{Float64,Float64}) where {BXT,BTT}
    it, bt   = Bt(t)
    it_d, btd = Bt(t, BSplineKit.Derivative(1))  # N'(t)

    ix, bx = Bx(x)

    kt = BSplineKit.order(Bt)
    kx = BSplineKit.order(Bx)

    Nt, Nx = length(Bt), length(Bx)
    dVdc = zeros(Nt, Nx)
    for δx in eachindex(bx), δt in eachindex(btd)
        ii = it - δt + 1
        jj = ix - δx + 1
        if ii >= 1 && ii <= Nt && jj >= 1 && jj <= Nx
            dVdc[ii, jj] = btd[δt] * bx[δx]    # N'_i(t) * M_j(x)
        end
    end
    return reshape(dVdc, :, )
end
spline2D_coeff_derivatives_time((Bt, Bx)::Tuple{BTT,BXT}, tx::Vector{Float64}) where {BXT,BTT} = spline2D_coeff_derivatives_time((Bt, Bx), (tx[1], tx[2]))


# ∂w/∂c = N_i(t) * M'_j(x) 
function spline2D_coeff_derivatives_space((Bt, Bx)::Tuple{BTT,BXT}, (t, x)::Tuple{Float64,Float64}) where {BXT,BTT}
    it, bt = Bt(t)
    ix, bx = Bx(x)
    ix_d, bxd = Bx(x, BSplineKit.Derivative(1))  # M'(x)

    kt = BSplineKit.order(Bt)
    kx = BSplineKit.order(Bx)   

    Nt, Nx = length(Bt), length(Bx)
    dWdc = zeros(Nt, Nx)
    for δx in eachindex(bxd), δt in eachindex(bt)
        ii = it - δt + 1
        jj = ix_d - δx + 1
        if ii >= 1 && ii <= Nt && jj >= 1 && jj <= Nx
            dWdc[ii, jj] = bt[δt] * bxd[δx]
        end
    end
    return reshape(dWdc, :, )
end
spline2D_coeff_derivatives_space((Bt, Bx)::Tuple{BTT,BXT}, tx::Vector{Float64}) where {BXT,BTT} = spline2D_coeff_derivatives_space((Bt, Bx), (tx[1], tx[2]))

function vector_hessian(f, x)
    S = length(f(x))    
    D = length(x)
    out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
    return reshape(out, S, D, D)  # return a 3D array where out[i, j, k] = ∂²f_i / ∂x_j ∂x_k
end