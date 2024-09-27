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

