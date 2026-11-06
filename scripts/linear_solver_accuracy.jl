# The accuracy of the direct linear solve (`linear_solver = true`) against Newton, slab by slab.
#
# The runs are the ones of the `linear_solver` testset of `test/ansatz/tensor_bspline.jl`: five
# steps of `h = 0.05` on `Wave` (`D = 1`) and on the linear `VectorWave` (`D = 2`, `β = 0`), a
# degree-2 tensor B-spline ansatz on `(2, 4)` cells with three Gauss nodes per cell, and each of the
# three couplings. For each element type, problem and coupling the script prints, per slab `n`,
#
#   lin-nl   max|u_lin − u_nl| / (eps(T) · max(1, max|u_nl|))
#   nl-ref   max|u_nl − u_ref| / (eps(T) · max(1, max|u_ref|))
#   lin-ref  max|u_lin − u_ref| / (eps(T) · max(1, max|u_ref|))
#
# with `u_ref` the Newton solution of the same configuration in `Float64`, and then the largest
# final residual `‖F(x)‖₂` over the steps of each path. It writes nothing.
#
#     julia --startup-file=no --project=. --check-bounds=auto scripts/linear_solver_accuracy.jl

using MultisymplecticIntegrators
using SimpleSplines: Dirichlet, Free
using Printf: @printf, @sprintf

function problems(::Type{T}) where {T}
    span = (timespan = (zero(T), T(0.25)), timestep = T(0.05), xstep = T(0.25))
    t1 = Wave.problem(; T = T, span...)
    t2 = VectorWave.problem(; T = T, span...)
    build(density, D, t) = LagrangianPDEProblem(density;
        D = D, ics_function = t.ics_function, bcs_function = t.bcs_function,
        timespan = t.timespan, timestep = t.timestep,
        xspan = t.xspan, xstep = t.xstep, params = t.params)
    return (("Wave", build(Wave.density(), 1, t1)),
        ("VectorWave", build(VectorWave.vector_density(; β = 0.0), 2, t2)))
end

const COUPLINGS = (("CoefficientRestriction", CoefficientRestriction(), Dirichlet()),
    ("WeakInitialCondition", WeakInitialCondition(), Dirichlet()),
    ("WeakInitialAndBoundaryConditions", WeakInitialAndBoundaryConditions(), Free()))

function run(::Type{T}, problem_index, coupling_index, linear_solver) where {T}
    _, problem = problems(T)[problem_index]
    _, coupling, boundary = COUPLINGS[coupling_index]
    ansatz = TensorBSplineAnsatz(
        degree = 2, ncells = (2, 4), xspan = (zero(T), one(T)), n = 3,
        x_boundary = boundary)
    return integrate(problem, SpaceTimeGalerkin(ansatz, coupling; linear_solver))
end

function deviation(a, b, ::Type{T}) where {T}
    maximum(abs.(Float64.(a) .- Float64.(b))) /
    (eps(T) * max(1.0, maximum(abs, Float64.(b))))
end

for T in (Float32, Float64), p in 1:2, k in 1:3
    reference = run(Float64, p, k, false)
    nl = run(T, p, k, false)
    lin = run(T, p, k, true)
    @printf("%-7s %-10s %-32s\n", T, problems(T)[p][1], COUPLINGS[k][1])
    @printf("  %-8s %s\n", "slab", join((@sprintf("%7d", n) for n in 1:5), " "))
    for (label, a, b) in (("lin-nl", lin, nl), ("nl-ref", nl, reference),
        ("lin-ref", lin, reference))
        figures = (deviation(a.solution.u[n], b.solution.u[n], T) for n in 1:5)
        @printf("  %-8s %s\n", label, join((@sprintf("%7.1f", f) for f in figures), " "))
    end
    @printf("  residual  nl %.2e  lin %.2e  lin iterations %s\n",
        maximum(nl.stats.residuals), maximum(lin.stats.residuals), lin.stats.iterations)
end
