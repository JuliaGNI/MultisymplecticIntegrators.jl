# The fixed-boundary affine control of test FB8, at the first slab of entries 7, 9 and 10 of the
# generated family: the singular values of the slab's Jacobian and the pivoted QR's diagonal near the
# rank cut, the residual ratio that each choice of rank leaves (the floor), the conditioning, and the
# ratio a run reaches after one, two, three and five Newton iterations.
#
#     julia --startup-file=no --project=. scripts/fb8_floor.jl
using LinearAlgebra, Random
using MultisymplecticIntegrators, SimpleSolvers, GeometricIntegratorsBase,
      GeometricSolutions
import GeometricIntegratorsBase: GeometricIntegrator, cache, current, integrate!, reset!,
                                 solutionstep
import GeometricBase: ntime, timesteps, parameters
const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "test", "helpers", "problem_generator.jl"))
const M = MultisymplecticIntegrators

T = Float64
linear = [e.problem for e in generate_problems(T; rng = Xoshiro(SEED)) if e.potential == 2]
cut = M.fixed_boundary_qr_cut(T)
for form in (:abg, :tsr), i in (7, 9, 10)

    problem = linear[i]
    a = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = form, mode = :last_layer,
        xspan = M.xspan(problem), nq = (6, 4), ncells = 20, D = M.components(problem))
    method = SpaceTimeGalerkin(a, FixedBoundary(); initial_guess = Continuation())
    int = GeometricIntegrator(problem, method; solver = Newton(), max_iterations = 2,
        verbosity = 0)
    c = cache(int)
    p = parameters(problem)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    reset!(solstep, timesteps(solution)[1])
    M.slab_bottom!(c, c.coupling, problem, a, current(solstep))
    M.initial_guess!(current(solstep), nothing, p, int)
    x0 = copy(c.x)
    r(x) = (b = zeros(T, length(x)); M.residual!(b, x, nothing, p, int); b)
    r0 = r(x0)
    n = length(x0)
    J = zeros(T, n, n)
    for j in 1:n
        e = zeros(T, n)
        e[j] = 1
        J[:, j] = (r(x0 .+ e) .- r(x0 .- e)) ./ 2
    end
    d = randn(Xoshiro(1), n)
    affine = norm(r(x0 .+ d) .- r0 .- J * d) / norm(J * d)
    σ = svdvals(J)
    rel = σ ./ σ[1]
    F = svd(J)
    # the residual ratio the minimum-norm step leaves when it keeps the k largest directions
    ratio(k) = norm(r0 - F.U[:, 1:k] * (F.U[:, 1:k]' * r0)) / norm(r0)
    near = findall(x -> 1e-14 < x < 1e-9, rel)
    q = qr(J, ColumnNorm())
    dR = abs.(diag(q.R)) ./ abs(q.R[1, 1])
    println("$form entry $i: n = $n, affine defect = $(round(affine, sigdigits = 2)), ",
        "rank(σ>cut) = $(count(>(cut), rel)), rank(|Rᵢᵢ|>cut) = $(count(>(cut), dR))")
    println("   σ/σ₁ in (1e-14, 1e-9): ", [round(rel[k], sigdigits = 3) for k in near])
    println("   |Rᵢᵢ|/|R₁₁| in (1e-14, 1e-9): ",
        [round(x, sigdigits = 3) for x in dR if 1e-14 < x < 1e-9])
    for k in unique([count(>(1e-9), rel), near..., count(>(1e-14), rel)])
        println("   keep $k: ratio = $(round(ratio(k), sigdigits = 3))")
    end
    kept = rel[count(>(cut), rel)]
    println("   smallest kept σ/σ₁ = $(round(kept, sigdigits = 3)), ",
        "cond·eps = $(round(eps(T) / kept, sigdigits = 3)), ",
        "eps·σ₁·‖x₀‖/‖r₀‖ = $(round(eps(T) * σ[1] * norm(x0) / norm(r0), sigdigits = 3))")
    for iterations in (1, 2, 3, 5)
        s = integrate(problem, method; solver = Newton(), max_iterations = iterations,
            f_reltol = M.fixed_boundary_f_reltol(T), verbosity = 0).stats
        println("   max_iterations = $iterations: ratio per step = ",
            round.(s.residuals ./ s.initial_residuals, sigdigits = 3),
            ", iterations = ", s.iterations)
    end
end

# The census of every slab FB8 runs — `Wave` and the linear entries, both forms, both element
# types: the rank and the floor that the SVD rule (singular values above `cut σ₁`) and the pivoted
# QR rule (the leading run of `|Rᵢᵢ|` above `cut |R₁₁|`, as `LapackPivotedQR` counts it) give.
function census(T)
    cut = M.fixed_boundary_qr_cut(T)
    wave = M.Wave.problem(T = T, timespan = (zero(T), T(0.05)), timestep = T(0.05),
        xstep = T(0.05), xspan = (zero(T), one(T)))
    family = [e.problem
              for e in generate_problems(T; rng = Xoshiro(SEED)) if e.potential == 2]
    for form in (:abg, :tsr), (i, problem) in enumerate(vcat([wave], family))

        a = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = form, mode = :last_layer,
            xspan = M.xspan(problem), nq = (6, 4), ncells = 20, D = M.components(problem))
        method = SpaceTimeGalerkin(a, FixedBoundary(); initial_guess = Continuation())
        int = GeometricIntegrator(problem, method; solver = Newton(), max_iterations = 2,
            f_reltol = M.fixed_boundary_f_reltol(T), verbosity = 0)
        c = cache(int)
        p = parameters(problem)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        for n in 1:ntime(problem)
            reset!(solstep, timesteps(solution)[n])
            M.slab_bottom!(c, c.coupling, problem, a, current(solstep))
            M.initial_guess!(current(solstep), nothing, p, int)
            r(x) = (b = zeros(T, length(x)); M.residual!(b, x, nothing, p, int); b)
            x0 = copy(c.x)
            r0 = r(x0)
            J = similar(x0, length(r0), length(x0))
            for j in eachindex(x0)
                e = zero(x0)
                e[j] = one(T)
                J[:, j] = (r(x0 .+ e) .- r(x0 .- e)) ./ 2
            end
            F = svd(J)
            ksvd = count(>(cut * F.S[1]), F.S)
            q = qr(J, ColumnNorm())
            d = abs.(diag(q.R))
            kqr = something(findfirst(≤(cut * d[1]), d), length(d) + 1) - 1
            Q = Matrix(q.Q)[:, 1:kqr]
            Usvd = F.U[:, 1:ksvd]
            floor_svd = norm(r0 - Usvd * (Usvd' * r0)) / norm(r0)
            floor_qr = norm(r0 - Q * (Q' * r0)) / norm(r0)
            label = i == 1 ? "wave" : "entry $(i - 1)"
            println("$T $form $label slab $n: n = $(length(x0)), rank svd = $ksvd, ",
                "rank qr = $kqr, floor svd = $(round(floor_svd, sigdigits = 3)), ",
                "floor qr = $(round(floor_qr, sigdigits = 3))")
            integrate!(solstep, int)
        end
    end
end
census(Float64)
census(Float32)
