struct PhysicalDomain
    domain::Vector{Tuple{Float64, Float64}}  # [(x1, x2), (y1, y2), ...] for d dimensions
    Δx::Float64 
    time_span::Tuple{Float64, Float64}  # (t1, t2)
    Δt::Float64

    function PhysicalDomain(domain::Vector{Tuple{Float64, Float64}}, Δx::Float64, time_span::Tuple{Float64, Float64}, Δt::Float64)
        new(domain, Δx, time_span, Δt)
    end
end

struct SolutionCache
    history::Vector{Array{Float64, 1}}
    current_step::Int
end

function initialize_cache(nt, nx)
    SolutionCache([zeros(nx) for _ in 1:nt],0)
end

parameter(name::Symbol) = Num(Sym{Real}(name))

using QuadratureRules
using IterTools

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


struct SolutionStepLPDE{TT,UT,VT,WT,XT}
    t::TT
    history

    u::UT
    v::VT
    w::WT   
    x::XT

    ū::UT
    v̄::VT
    w̄::WT
    x̄::XT

    params::NamedTuple
    internal
    function SolutionStepLPDE(t::TT,u::UT,v::VT,w::WT,x,::XT,params::PT; nhistory = 2,internal::IT = NamedTuple()) where {TT,UT,VT,WT,XT,PT,IT}
        @assert nhistory ≥ 1 "nhistory must be greater than 0"

        history = (
            t = OffsetVector([zero(t) for _ in 0:nhistory], 0:nhistory),
            u = OffsetVector([zero(u) for _ in 0:nhistory], 0:nhistory),
            v = OffsetVector([zero(v) for _ in 0:nhistory], 0:nhistory),
            w = OffsetVector([zero(w) for _ in 0:nhistory], 0:nhistory),
            x = OffsetVector([zero(x) for _ in 0:nhistory], 0:nhistory),
        )

        u = history.u[0]
        v = history.v[0]
        w = history.w[0]
        x = history.x[0]

        ū = history.u[1]
        v̄ = history.v[1]
        w̄ = history.w[1]
        x̄ = history.x[1]

        return new{typeof(t),typeof(u),typeof(v),typeof(w),typeof(x)}(t, history, u, v, w, x, ū, v̄, w̄, x̄, params, internal)
    end
end