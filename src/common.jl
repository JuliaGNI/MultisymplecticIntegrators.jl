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

