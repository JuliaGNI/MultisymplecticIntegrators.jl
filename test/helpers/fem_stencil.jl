using LinearAlgebra: lu

"""
    fem_stencil(problem; steps)

The space-time finite-element stencil of the wave equation at the lowest order, as an oracle that is
written out as a recurrence rather than assembled by the integrator. On every slab its discrete
equations are

```math
M_x \\,(u^{n+1} - 2 u^n + u^{n-1}) = -\\frac{h^2 c^2}{6} \\, K_x \\,(u^{n-1} + 4 u^n + u^{n+1}),
```

with the linear-element matrices `M_x = (Δx / 6) (1, 4, 1)` and `K_x = (-1, 2, -1) / Δx`. The first
slab carries the initial momentum as a load, `(M_x + s K_x) u^1 = (M_x - 2 s K_x) u^0 + h l^0` with
`s = h^2 c^2 / 6` and `l^0_j = ∫ v^0 φ_j dx` taken by the two-point Gauss rule per cell; every later
slab is the recurrence above.

It is the bilinear element that the §A values of `FEM_Multisymplectic_Integrator` were measured
with. Returned are the grid and one row of values per step.
"""
function fem_stencil(problem; steps::Integer)
    T = datatype(problem)
    c = problem.params.c
    h = problem.timestep
    Δx = problem.xstep
    grid = collect(xgrid(problem))
    N = length(grid)
    interior = 2:(N - 1)
    M, K = fem_matrices(problem)
    # the load ∫ v⁰ φ_j dx, by the two-point Gauss rule of each cell
    ξ = sqrt(one(T) / 3)
    nodes = T[]
    weights = T[]
    for cell in 1:(N - 1)
        a, b = grid[cell], grid[cell + 1]
        for σ in (-ξ, ξ)
            push!(nodes, a + (b - a) * (one(T) + σ) / 2)
            push!(weights, (b - a) / 2)
        end
    end
    velocity = problem.ics_function(nodes, problem.params).v
    load = zeros(T, N)
    for (r, node) in enumerate(nodes), j in interior

        φ = max(zero(T), one(T) - abs((node - grid[j]) / Δx))
        load[j] += weights[r] * velocity[r] * φ
    end
    initial = copy(problem.ics_function(grid, problem.params).u)
    s = h^2 * c^2 / 6
    lhs = M + s * K
    factor = lu(lhs[interior, interior])
    result = zeros(T, steps, N)
    # the first slab carries the initial momentum as a load, not a previous state
    current = copy(initial)
    current[interior] = factor \ ((M - 2s * K) * initial + h * load)[interior]
    previous = initial
    result[1, :] .= current
    for n in 2:steps
        rhs = (2M - 4s * K) * current - lhs * previous
        next = copy(current)
        next[interior] = factor \ rhs[interior]
        previous, current = current, next
        result[n, :] .= current
    end
    return grid, result
end

"""
    fem_matrices(problem)

The linear-element matrices of the stencil, `M_x = (Δx / 6) (1, 4, 1)` and `K_x = (-1, 2, -1) / Δx`,
on the whole grid. The rows of the interior grid points carry the three-point stencil and the two
boundary rows are zero, so a product is the interior part of the discrete operator.
"""
function fem_matrices(problem)
    T = datatype(problem)
    Δx = problem.xstep
    N = length(xgrid(problem))
    M = zeros(T, N, N)
    K = zeros(T, N, N)
    for j in 2:(N - 1)
        M[j, j - 1], M[j, j], M[j, j + 1] = Δx / 6, 4Δx / 6, Δx / 6
        K[j, j - 1], K[j, j], K[j, j + 1] = -1 / Δx, 2 / Δx, -1 / Δx
    end
    return M, K
end
