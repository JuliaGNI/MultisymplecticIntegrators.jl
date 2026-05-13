struct OGA2D <: InitialParametersMethod 
    equispaced_quad_nodes::Matrix{Float64}  # 2 × N
    quad_weights::Vector{Float64}            # N
    A_mat::Matrix{Float64}                   # M × 3
    Φ_raw::Matrix{Float64}                      # M × N
    N::Int
    M::Int
    nx::Int
    nt::Int
    Nw::Int
    Nb::Int

    function OGA2D(a::Float64,b::Float64,activation::Function; nx::Int = 40, nt::Int= 20, Nw::Int=500, Nb::Int=500)
        # Equidistant Quadrature / sampling grid
        xs = range(a, b, length=nx)
        ts = range(0.0, 1.0, length=nt)

        # build list of sample coords as 2×N matrix (t; x)
        coords = [(t, x) for t in ts, x in xs]   # nt × nx array of tuples
        N = length(coords)
        equispaced_quad_nodes = zeros(2, N)
        for i in 1:N
            equispaced_quad_nodes[1, i] = coords[i][1]
            equispaced_quad_nodes[2, i] = coords[i][2]
        end

        # simple uniform quadrature weights (you can switch to Simpson)
        quad_weights = fill(1.0 / N, N)
        thetas = range(-π, π, length=Nw + 1)
        dirs = [[cos(θ), sin(θ)] for θ in thetas]  # length Nw+1

        biases = range(-π, π, length=Nb + 1)       # larger bias range works well for sinusoids

        # make dictionary rows (M × 3)
        M = length(dirs) * length(biases)
        A_mat = Matrix{Float64}(undef, M, 3)
        idx = 1
        for w in dirs, b in biases
            A_mat[idx, 1] = w[1]
            A_mat[idx, 2] = w[2]
            A_mat[idx, 3] = b
            idx += 1
        end

        # build augmented coordinates (for bias): 3 × N
        Xaug = vcat(equispaced_quad_nodes, ones(1, N))

        # precompute dictionary activations (M×N)
        Φ_raw = activation.(A_mat * Xaug)   # M × N
        new(equispaced_quad_nodes, quad_weights, A_mat, Φ_raw, N, M, nx, nt, Nw, Nb)
    end

end