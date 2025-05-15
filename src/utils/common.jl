function construct_quadrature_grid(dimensions::Vector{Int})
    # Create quadrature rules for each dimension
    quadrature_rules = [QuadratureRules.GaussLegendreQuadrature(R) for R in dimensions]

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

GeometricIntegrators.Integrators.nlsolution(c::PDEIntegratorCache) = c.x

Lagrangian_multiplier(::Val{:BSplineDirichlet},order::Integer,quad_nodes::Vector{Float64}) = BSplineDirichlet(order,quad_nodes)
Lagrangian_multiplier(::Val{:Lagrange},order::Integer,quad_nodes::Vector{Float64}) = CompactBasisFunctions.Lagrange(quad_nodes)
function Lagrangian_multiplier(basis::Symbol, order::Integer, quad_nodes::Vector{Float64})
    if basis ∉ (:BSplineDirichlet, :Lagrange)
        error("Unsupported basis: $basis")
    end
    Lagrangian_multiplier(Val(basis), order, quad_nodes)
end