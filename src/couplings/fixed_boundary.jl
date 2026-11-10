# The fixed-boundary coupling: the datum of one slab, the momentum the slab hands on, the fit of the
# method's own guess and the predictor it starts from.
#
# The network block of this method holds the coupling's fit workspace, and the coupling's methods
# dispatch on the block, so the two files refer to each other. The module therefore includes this
# file twice: the first pass, before `ansatz/network.jl`, declares the two blocks and the fits, and
# the second pass, after it, declares the methods that dispatch on the network's ansatz and block.
# The guard below is what tells the passes apart.

if !@isdefined(FixedBoundaryData)
    """
        FixedBoundary() <: SlabCoupling

    The coupling of the fixed-boundary network: it carries the datum of a slab — the previous
    level's parameters `φ₀` and the momentum `p₀` that level received — and hands the momentum of the
    solved level on to the next slab. It has no multiplier and no equation of its own: the step's
    equations are the action's stationarity alone.

    A `FixedBoundary` takes a [`FixedBoundaryNetwork`](@ref) and nothing else, and that ansatz takes
    this coupling and nothing else; either mismatch raises `ArgumentError` when the integrator is
    built.
    """
    struct FixedBoundary <: SlabCoupling end

    """
        FixedBoundaryData{ST} <: CouplingData

    The block of a [`FixedBoundary`](@ref) in the slab cache.

    `phi` and `momentum` are the datum's pair `(φ₀, p₀)`, the previous level's parameters and the
    covector on the free ones. `ga` and `gb` hold the two ends' side data, one row per time of the
    slab: the slab's start time in row 1, its nodes in rows `2:(qt + 1)`, and its top time in row
    `qt + 2`; `dga` and `dgb` hold the same rows' derivatives in physical time. `start` is the
    slab's start time.

    `phi_next` is the solved level's parameters and `p_next` the momentum the step hands on; the
    next slab's datum is filled from these two copies. `v_top` is the solved slab's velocity at its
    top time, on the predictor's grid, which a later slab's predictor starts from. All three are
    taken when the step's solution is written, so that no buffer a later residual or a later guess
    rewrites can move them.
    """
    mutable struct FixedBoundaryData{ST} <: CouplingData
        coupling::FixedBoundary
        phi::Vector{ST}
        momentum::Vector{ST}
        ga::Matrix{ST}
        gb::Matrix{ST}
        dga::Matrix{ST}
        dgb::Matrix{ST}
        start::ST
        p_next::Vector{ST}
        phi_next::Vector{ST}
        v_top::Matrix{ST}
    end

    """
        BubbleGreedy(; nα = 13, nβ = 13, nγ = 13, α_range = (-8.0, 8.0), β_range = (-8.0, 8.0),
                      γ_range = (-6.0, 6.0), iterations = 60)

    The fit that starts a [`FixedBoundaryNetwork`](@ref) slab: the level's parameters by
    Levenberg–Marquardt from the datum to the target at the top of the slab, and then the bubble by a
    greedy selection of `St` atoms out of a dictionary of `tanh(α t̂ + β ŝ + γ)` units.

    The dictionary is `nα × nβ × nγ` atoms on the three ranges, less the atoms at `α = β = 0`, which
    no `(θ, s, ρ)` writes; both forms of the bubble therefore start from the same atoms, and the
    units are read in the form's own coordinates. The selection takes, at every step, the atom whose
    correlation with the target's remainder is largest in absolute value over the atoms and the
    components, as the greedy fit of a network does, and its output weights are the least-squares
    solution on the atoms taken so far.
    """
    struct BubbleGreedy{R <: Tuple{Vararg{Real, 2}}} <: SlabFitMethod
        nα::Int
        nβ::Int
        nγ::Int
        α_range::R
        β_range::R
        γ_range::R
        iterations::Int
    end

    function BubbleGreedy(; nα::Integer = 13, nβ::Integer = 13, nγ::Integer = 13,
            α_range = (-8.0, 8.0), β_range = (-8.0, 8.0), γ_range = (-6.0, 6.0),
            iterations::Integer = 60)
        nα ≥ 1 && nβ ≥ 1 && nγ ≥ 1 || throw(ArgumentError(
            "the dictionary needs a positive number of atoms in each range, got ($nα, $nβ, $nγ)"))
        for (name, r) in (("α_range", α_range), ("β_range", β_range), ("γ_range", γ_range))
            length(r) == 2 && all(isfinite, r) && r[1] < r[2] || throw(ArgumentError(
                "$name must hold two increasing finite values, got $r"))
        end
        iterations ≥ 0 ||
            throw(ArgumentError("iterations must be at least 0, got $iterations"))
        # The three ranges are stored at the element type their own values promote to, so that a
        # dictionary of another precision is built from that precision's numbers and not from
        # `Float64`'s.
        E = float(promote_type(eltype(α_range), eltype(β_range), eltype(γ_range)))
        ranges = map(r -> (E(r[1]), E(r[2])), (α_range, β_range, γ_range))
        return BubbleGreedy(Int(nα), Int(nβ), Int(nγ), ranges..., Int(iterations))
    end

    """
        bubble_atoms(method::BubbleGreedy, ::Type{T})

    The dictionary of a bubble fit, as `(α, β, γ)` rows of precision `T`: every combination of the
    method's three ranges, less the atoms at `α = β = 0`, which have no `(θ, s, ρ)` coordinates and
    are dropped so that both forms of the bubble start from the same units.
    """
    function bubble_atoms(method::BubbleGreedy, ::Type{T}) where {T}
        α = range(method.α_range[1], method.α_range[2]; length = method.nα)
        β = range(method.β_range[1], method.β_range[2]; length = method.nβ)
        γ = range(method.γ_range[1], method.γ_range[2]; length = method.nγ)
        atoms = Matrix{T}(undef, method.nα * method.nβ * method.nγ, 3)
        k = 0
        for c in γ, b in β, a in α
            k += 1
            atoms[k, 1], atoms[k, 2], atoms[k, 3] = T(a), T(b), T(c)
        end
        keep = [i for i in axes(atoms, 1) if !(iszero(atoms[i, 1]) && iszero(atoms[i, 2]))]
        return atoms[keep, :]
    end

    """
        FixedBoundaryFit{ST}

    The workspace of the fit that starts a fixed-boundary slab: the dictionary its method builds, that
    dictionary tabulated at the slab's nodes and weighted by `√(ωₜωₓ)`, the buffers of the greedy
    selection over the atoms and the components, and the output weights it leaves.

    The workspace is keyed on the method's own parameters as well as on the sizes, so that a second
    fit of the same size reads its own dictionary and not the first one's.
    """
    mutable struct FixedBoundaryFit{ST}
        method::Union{Nothing, BubbleGreedy}
        atoms::Matrix{ST}
        table::Matrix{ST}
        target::Matrix{ST}
        chosen::Matrix{ST}
        gram::Matrix{ST}
        correlations::Matrix{ST}
        residual::Matrix{ST}
        coefficients::Matrix{ST}
        selected::Vector{Int}
        taken::Vector{Bool}
        weighted::Vector{ST}
    end

    function FixedBoundaryFit(::Type{ST}) where {ST}
        empty2 = zeros(ST, 0, 0)
        return FixedBoundaryFit{ST}(
            nothing, empty2, empty2, empty2, empty2, empty2, empty2,
            empty2, empty2, Int[], Bool[], zeros(ST, 0))
    end

    """
        Predictor() <: SlabTarget

    A cheap predictor of the slab, as the target of a `FitTo`: Störmer–Verlet on a uniform grid of
    the span, from the bottom level's field and either the initial data's velocity or the previous
    slab's top velocity, with the side data imposed at every substep. The guess fits the level to the
    predictor's top and the bubble to what the predictor has beyond the interpolation of the two
    levels.

    It is a target of a [`FixedBoundaryNetwork`](@ref) alone, and its acceleration reads the density's
    velocity derivative as one constant `m` times the velocity — which holds for the wave equation,
    for sine-Gordon and for a generated density, and fails for a density whose momentum is not of
    that form, such as a doubled linear transport. Either mismatch raises `ArgumentError` when the
    integrator is built.
    """
    struct Predictor <: SlabTarget end

    """The number of grid points of the predictor's space stencil."""
    const PREDICTOR_POINTS = 101

    """The number of Störmer–Verlet substeps of the predictor."""
    const PREDICTOR_SUBSTEPS = 10

    """
        fixed_boundary_qr_cut(::Type{T})

    The relative rank cut of the pivoted QR a `Newton()` step takes, at element type `T`.

    `Float64` is `1e-12`. The factorization's own default `√eps(T) = 1.49e-8` drops two singular
    directions of a slab's Jacobian in which the residual stalls, so the cut is the floor the step
    reaches rather than the factorization's guess. Another element type takes `10 eps(T)`, the same
    floor in its own round-off: a cut below it would keep directions that the factorization at that
    precision cannot resolve.
    """
    fixed_boundary_qr_cut(::Type{Float64}) = 1e-12
    fixed_boundary_qr_cut(::Type{T}) where {T} = 10 * eps(T)

    """
        fixed_boundary_f_reltol(::Type{T})

    The relative tolerance of a fixed-boundary step's convergence rule, at element type `T`.

    `Float64` is `1e-8`. On `Wave` at `h = 0.05` the dead branch's twenty steps reach a worst ratio of
    `2.96e-10` under this rule, with the rank cut of the pivoted QR at `1e-12`; the tolerance is not
    what limits that run, since the factorization's own cut leaves one step at `3.0e-8`.
    `Float32` is `3.0f-2`, the value measured on the family's affine control at that element type,
    whose worst ratio is `1.365e-2`; it is never the `Float64` value converted. Any other element
    type is refused rather than given the floor of its own round-off: the rule is a measurement on
    the family's control and not a formula, so a run at another type names the tolerance it is
    solved at as an option of its own, and a run that names none is told the type it asked for.
    """
    fixed_boundary_f_reltol(::Type{Float64}) = 1e-8

    fixed_boundary_f_reltol(::Type{Float32}) = 3.0f-2

    function fixed_boundary_f_reltol(::Type{T}) where {T}
        throw(ArgumentError("the fixed-boundary network has no measured f_reltol at $T; " *
                            "name the tolerance as an option of its own"))
    end

    """
        resolved_f_reltol(options, ::Type{T})

    The solver options with their `f_reltol` filled in: the caller's own where it names one, and
    [`fixed_boundary_f_reltol`](@ref) at the element type where it does not. [`default_options`](@ref)
    hands the tolerance on as `nothing` rather than resolving it there, because
    `GeometricIntegrator` merges the default bundle into the caller's keywords before any solver is
    built: only here is a caller's own `f_reltol` known, and only a caller that names none reaches
    the element type's measured value. Options with no `f_reltol` key at all belong to a method of
    another family and come back as they are.
    """
    function resolved_f_reltol(options::NamedTuple, ::Type{T}) where {T}
        haskey(options, :f_reltol) || return options
        options.f_reltol === nothing || return options
        return merge(options, (f_reltol = fixed_boundary_f_reltol(T),))
    end

else
    # ---- the second pass: the methods that dispatch on the network's ansatz and block ----------

    """
        coupling_data(coupling, ansatz, problem, ST)

    The coupling's block of the slab cache: the datum's arrays of the problem's element type and
    size, filled by `slab_bottom!` at every slab.
    """
    function coupling_data(c::FixedBoundary, a::FixedBoundaryNetwork,
            ::LagrangianPDEProblem, ::Type{ST}) where {ST}
        qt, D = length(a.tabulation.tnode), a.D
        side() = zeros(ST, qt + 2, D)
        nfp = length(spatial_free(a))
        return FixedBoundaryData{ST}(c, zeros(ST, nspatial(a)), zeros(ST, nfp),
            side(), side(), side(), side(), zero(ST), zeros(ST, nfp),
            zeros(ST, nspatial(a)), zeros(ST, PREDICTOR_POINTS, D))
    end

    function coupling_data_type(c::FixedBoundary, a::FixedBoundaryNetwork,
            ::LagrangianPDEProblem, ::Type{ST}) where {ST}
        return FixedBoundaryData{ST}
    end

    """The datum of a slab reaches the field through this function, as `slab_field!`'s `bottom`."""
    bottom_datum(data::FixedBoundaryData) = data

    """The solver's vector holds the level's free parameters and the bubble's, with no multiplier."""
    free_rows(::FixedBoundary, a::FixedBoundaryNetwork) = nfree(a)

    """A fixed-boundary coupling has no unknown of its own."""
    coupling_unknowns(::FixedBoundary, ::FixedBoundaryNetwork, ::LagrangianPDEProblem) = 0

    """
        coupling_term!(b, x, data, cache, ansatz, problem)

    The datum's momentum adds to the first block of the action's gradient, which the ansatz wrote on
    the equations of the solved level: the step's residual is `(p₀ + ∂S/∂φ₀, ∂S/∂Θ)`, and the
    coupling has no other equation.
    """
    function coupling_term!(b::AbstractVector, x::AbstractVector, data::FixedBoundaryData,
            c::SpaceTimeGalerkinCache, ansatz::FixedBoundaryNetwork,
            problem::LagrangianPDEProblem)
        for k in eachindex(data.momentum)
            b[k] += data.momentum[k]
        end
        return nothing
    end

    """A fixed-boundary coupling guesses no unknown of its own: the datum is the coupling's guess."""
    function coupling_guess!(::FixedBoundaryData, cache, ansatz::SpaceTimeAnsatz,
            problem::LagrangianPDEProblem, sol)
        return nothing
    end

    """
        coupling_update!(data, x, cache, ansatz, problem)

    Keep what the solved slab hands on to the next one: the momentum the step passes on, the level
    the step produced, and that slab's velocity at its top time on the predictor's grid, each as a
    copy taken now. The block's own `params` and `momentum` are read after the solve and before any
    later guess overwrites them, so a copy is the only way to keep them.
    """
    function coupling_update!(data::FixedBoundaryData, x::AbstractVector,
            c::SpaceTimeGalerkinCache, ansatz::FixedBoundaryNetwork,
            problem::LagrangianPDEProblem)
        copyto!(data.p_next, c.ansatz.momentum)
        copyto!(data.phi_next, 1, c.ansatz.params, 1, nspatial(ansatz))
        top_velocity!(data, c.ansatz, ansatz, problem)
        return nothing
    end

    """
        top_velocity!(data, s, ansatz, problem)

    The velocity of the solved slab's field at its top time, `∂_{t̂}u / h`, at the predictor's grid:
    the field's time derivative from the slab's own datum `φ₀`, the level the solve produced, the
    bubble it produced, and the side data at its two ends. This is the velocity the next slab's
    predictor starts from, because the next slab's datum is the level this one produced and nothing
    of this slab's bottom level or bubble survives into it.
    """
    function top_velocity!(data::FixedBoundaryData{ST}, s::FixedBoundarySlab{ST},
            ansatz::FixedBoundaryNetwork, problem::LagrangianPDEProblem) where {ST}
        a = ansatz
        h = convert(ST, timestep(problem))
        φ0 = data.phi
        φ1 = view(data.phi_next, 1:nspatial(a))
        Θ = bubble_parameters(s)
        for (r, x) in enumerate(predictor_grid(a, ST)), d in axes(data.v_top, 2)

            _, ut, _ = slab_component(a, d, φ0, φ1, Θ, one(ST), x, h,
                top_sides(data, d))
            data.v_top[r, d] = ut / h
        end
        return nothing
    end

    """Copy the datum of the problem's element type into the cache the solver works in."""
    function copy_coupling_data!(c::FixedBoundaryData, data::FixedBoundaryData)
        copyto!(c.phi, data.phi)
        copyto!(c.momentum, data.momentum)
        c.ga .= data.ga
        c.gb .= data.gb
        c.dga .= data.dga
        c.dgb .= data.dgb
        c.start = data.start
        copyto!(c.p_next, data.p_next)
        copyto!(c.phi_next, data.phi_next)
        c.v_top .= data.v_top
        return nothing
    end

    """
        check_coupling(problem, coupling::FixedBoundary, ansatz)

    Refuse either half of the pair on its own: the coupling takes a [`FixedBoundaryNetwork`](@ref)
    and the ansatz takes this coupling, and each refusal names the partner the method accepts.
    """
    function check_coupling(problem::LagrangianPDEProblem, c::FixedBoundary,
            a::SpaceTimeAnsatz)
        a isa FixedBoundaryNetwork || throw(ArgumentError(
            "FixedBoundary takes a FixedBoundaryNetwork, got $(nameof(typeof(a)))"))
        return nothing
    end

    function check_coupling(problem::LagrangianPDEProblem, c::FixedBoundary,
            a::FixedBoundaryNetwork)
        return nothing
    end

    function check_coupling(problem::LagrangianPDEProblem, c::SlabCoupling,
            a::FixedBoundaryNetwork)
        throw(ArgumentError(
            "a FixedBoundaryNetwork takes the FixedBoundary coupling, got $(nameof(typeof(c)))"))
    end

    """A fixed-boundary slab is started by the fit of the bubble."""
    default_fit(::FixedBoundaryNetwork) = BubbleGreedy()

    """A fixed-boundary slab is solved by Levenberg–Marquardt."""
    default_solver(::SpaceTimeGalerkin{<:FixedBoundaryNetwork}) = LevenbergMarquardt()

    """
        default_options(method, problem)

    The family's solver options: a step converges on U12's rule at `f_abstol = 0` and the element
    type's own relative tolerance, and a solve is capped at 300 iterations. They are the defaults of
    both the family's solver and the `Newton()` a caller may name instead, so that an affine control
    and a dead branch are measured under one rule.

    The tolerance is handed on as `nothing`, because `GeometricIntegrator` merges these defaults
    into the caller's keywords before any solver is built: the family's `initsolver` resolves it from
    [`fixed_boundary_f_reltol`](@ref) once the caller's own keywords are known, so that a caller who
    names an `f_reltol` at an element type the family has not measured is not refused for it.
    """
    function default_options(
            method::SpaceTimeGalerkin{<:FixedBoundaryNetwork},
            problem::LagrangianPDEProblem)
        T = datatype(problem)
        return (f_abstol = zero(T), f_reltol = nothing,
            max_iterations = LM_MAX_ITERATIONS)
    end

    """
        initsolver(solvermethod::NonlinearSolverMethod, method, caches; kwargs...)

    A fixed-boundary integrator's solver, built from the options the integrator was handed: the
    family's tolerance is resolved here, in the one method every nonlinear solver of the family
    reaches.

    [`default_options`](@ref) hands `f_reltol` on as `nothing`, because `GeometricIntegrator` merges
    those defaults into the caller's keywords before any solver is built. Here the caller's own
    keywords are known: a caller who names a tolerance keeps it, and one who names none is solved at
    [`fixed_boundary_f_reltol`](@ref) of the problem's element type, whichever nonlinear solver it
    named.
    """
    function initsolver(solvermethod::NonlinearSolverMethod,
            method::SpaceTimeGalerkin{<:FixedBoundaryNetwork}, caches::CacheDict; kwargs...)
        options = resolved_f_reltol(values(kwargs), eltype(nlsolution(caches)))
        return invoke(initsolver,
            Tuple{NonlinearSolverMethod, GeometricMethod, CacheDict},
            solvermethod, method, caches; options...)
    end

    """
        initsolver(solvermethod::LevenbergMarquardt, method, caches; kwargs...)

    The family's default solver, its tolerance resolved as above.

    It is a method of its own because the generic Levenberg–Marquardt method and the family's method
    for every nonlinear solver would otherwise both apply to this method and this solver, and this
    one is the more specific of the three.
    """
    function initsolver(solvermethod::LevenbergMarquardt,
            method::SpaceTimeGalerkin{<:FixedBoundaryNetwork}, caches::CacheDict; kwargs...)
        options = resolved_f_reltol(values(kwargs), eltype(nlsolution(caches)))
        return invoke(initsolver, Tuple{LevenbergMarquardt, GeometricMethod, CacheDict},
            solvermethod, method, caches; options...)
    end

    """
        initsolver(solvermethod::Newton, method, caches; kwargs...)

    A fixed-boundary slab's rows are the derivative of a discrete action, so its Jacobian is a
    saddle point whose rank is below its size: `Newton()` takes the minimum-norm step of a complete
    orthogonal factorization, as the multiplier network does.

    The factorization's rank cut is [`fixed_boundary_qr_cut`](@ref), which the default cut `√eps(T)`
    would leave the step's residual stalled above. A caller's own `linear_solver_method` still
    overrides it.
    """
    function initsolver(
            solvermethod::Newton, method::SpaceTimeGalerkin{<:FixedBoundaryNetwork},
            caches::CacheDict; kwargs...)
        T = eltype(nlsolution(caches))
        options = resolved_f_reltol(
            merge(
                (;
                    linear_solver_method = SimpleSolvers.LapackPivotedQR(
                    rtol = fixed_boundary_qr_cut(T))),
                values(kwargs)),
            T)
        return invoke(initsolver,
            Tuple{NonlinearSolverMethod, GeometricMethod, CacheDict},
            solvermethod, method, caches; options...)
    end

    """
        variable_projection_indices(method::SpaceTimeGalerkin{<:FixedBoundaryNetwork})

    The positions of the bubble's output weights in the solver's vector, which the projected variant
    eliminates by an inner solve.
    """
    function variable_projection_indices(
            method::SpaceTimeGalerkin{<:FixedBoundaryNetwork})
        a = ansatz(method)
        nfp = length(spatial_free(a))
        nb = length(bubble_free(a))
        return collect((nfp + nb - a.St * a.D + 1):(nfp + nb))
    end

    # ---- the predictor's density ---------------------------------------------------------

    # One component's `∂L/∂v` at a point, whatever the density's shape.
    @inline function velocity_derivative(density::LagrangianDensity, u, v, w, params,
            d::Integer)
        return derivative_v(density, u[d], v[d], w[d], params)
    end

    @inline function velocity_derivative(density, u, v, w, params, d::Integer)
        return derivatives(density, u, v, w, params).v[d]
    end

    @inline function field_derivative_u(density::LagrangianDensity, u, v, w, params,
            d::Integer)
        return derivative_u(density, u[d], v[d], w[d], params)
    end

    @inline function field_derivative_u(density, u, v, w, params, d::Integer)
        return derivatives(density, u, v, w, params).u[d]
    end

    @inline function field_derivative_w(density::LagrangianDensity, u, v, w, params,
            d::Integer)
        return derivative_w(density, u[d], v[d], w[d], params)
    end

    @inline function field_derivative_w(density, u, v, w, params, d::Integer)
        return derivatives(density, u, v, w, params).w[d]
    end

    # The two points at which the predictor reads the density's velocity derivative, the same points
    # for every problem so that a refusal does not depend on a draw.
    function predictor_points(::Type{T}, D::Integer) where {T}
        (
            (fill(T(37) / T(100), D), fill(T(-23) / T(100), D)),
            (fill(T(-51) / T(100), D), fill(T(44) / T(100), D)))
    end

    """
        predictor_mass(density, params, T, D)

    The constant `m` of a density whose velocity derivative is `∂L/∂v = m v`, one per component:
    the ratio `∂L/∂v / v`, read at two seeded points and at two nonzero velocities.

    A density whose `∂L/∂v` does not vanish at zero velocity, whose ratio is zero, or whose ratio
    is not the same at the two velocities — as `m v³` or `m v²` give — has no such constant, and a
    predictor on it would integrate a different equation than the slab solves: it raises
    `ArgumentError` here, when the integrator is built.
    """
    function predictor_mass(density, params, ::Type{T}, D::Integer) where {T}
        mass = fill(convert(T, NaN), D)
        for (u, w) in predictor_points(T, D)
            for v in (zero(T), one(T), T(2))
                for d in 1:D
                    m = velocity_derivative(density, u, fill(v, D), w, params, d)
                    if iszero(v)
                        abs(m) ≤ 4 * eps(T) || throw(ArgumentError(
                            "Predictor() needs a density whose ∂L/∂v vanishes at zero velocity; " *
                            "component $d of $(nameof(typeof(density))) gives $m there"))
                    else
                        ratio = convert(T, m / v)
                        (isnan(mass[d]) || mass[d] == ratio) || throw(ArgumentError(
                            "Predictor() needs one constant ∂L/∂v / v for each velocity of a " *
                            "component; $(nameof(typeof(density))) gives $(mass[d]) and $ratio"))
                        iszero(ratio) && throw(ArgumentError(
                            "Predictor() needs a density whose ∂L/∂v is not zero; component $d " *
                            "of $(nameof(typeof(density))) gives $m"))
                        mass[d] = ratio
                    end
                end
            end
        end
        return mass
    end

    function check_predictor_density(problem::LagrangianPDEProblem)
        predictor_mass(problem.density, parameters(problem), datatype(problem),
            components(problem))
        return nothing
    end

    # ---- the datum of a slab -------------------------------------------------------------

    # The two ends' side data of one time, and its derivative in physical time, into the row of the
    # coupling's block that holds it.
    function side_row!(data::FixedBoundaryData{ST}, problem::LagrangianPDEProblem,
            row::Integer, t) where {ST}
        p = parameters(problem)
        bcs = problem.bcs_function(t, p)
        for d in eachindex(data.ga[row, :])
            data.ga[row, d] = ST(bcs.left.u[d])
            data.gb[row, d] = ST(bcs.right.u[d])
            data.dga[row, d] = ST(side_derivative(
                τ -> problem.bcs_function(τ, p).left.u[d], t))
            data.dgb[row, d] = ST(side_derivative(
                τ -> problem.bcs_function(τ, p).right.u[d], t))
        end
        return nothing
    end

    # One component of a level's network at one point, from the side data at that level's time.
    function spatial_network(a::FixedBoundaryNetwork, φ::AbstractVector, x, ga, gb,
            d::Integer)
        span = a.xspan[2] - a.xspan[1]
        ŝ = (x - a.xspan[1]) / span
        m, _ = spatial_values(φ, a.Sx, a.D, d, ŝ)
        return (1 - ŝ) * ga + ŝ * gb + (x - a.xspan[1]) * (a.xspan[2] - x) * m
    end

    """
        first_datum!(data, cache, problem, ansatz, sol)

    Fix the datum of a run's first slab: `φ₀` is a least-squares fit of the level's network to the
    initial condition at the nodes of the space rule, weighted by the square root of the rule's
    weights, and `p₀` is `∫ ∂L/∂v(u₀, v₀, w₀) · ∂N/∂φ dx` on that rule, on the parameters the mode
    frees.
    """
    function first_datum!(data::FixedBoundaryData{ST}, problem::LagrangianPDEProblem,
            a::FixedBoundaryNetwork, t₀) where {ST}
        nodes = a.tabulation.xnode
        weights = a.tabulation.xweight
        D = a.D
        params = parameters(problem)
        initial = problem.ics_function(collect(nodes), params)
        ga0, gb0 = data.ga[1, :], data.gb[1, :]
        # the fit of the level to u₀
        u₀ = ST.(initial.u)
        r!(b, φ) = begin
            k = 0
            for r in eachindex(nodes), d in 1:D

                k += 1
                b[k] = sqrt(weights[r]) *
                       (spatial_network(a, φ, nodes[r], ga0[d], gb0[d], d) - u₀[r, d])
            end
            b
        end
        φ = zeros(ST, nspatial(a))
        hidden_start!(φ, a, ST)
        least_squares!(φ, r!, length(nodes) * D; iterations = 1000,
            μ₀ = convert(ST, 1e-3))
        copyto!(data.phi, φ)
        # the momentum of the initial data on the free parameters
        free = spatial_free(a)
        fill!(data.momentum, zero(ST))
        for r in eachindex(nodes)
            u = Tuple(initial.u[r, :])
            v = Tuple(initial.v[r, :])
            w = Tuple(initial.w[r, :])
            g = gradient(
                z -> sum(velocity_derivative(problem.density, u, v, w, params, d) *
                         spatial_network(a, z, nodes[r], ga0[d], gb0[d], d) for d in 1:D),
                φ)
            for (k, i) in enumerate(free)
                data.momentum[k] += weights[r] * g[i]
            end
        end
        return nothing
    end

    # The hidden layer's starting point of the first slab's fit: the fixed-boundary spike's grid, so
    # that the fit starts where the reference implementation starts.
    function hidden_start!(φ::AbstractVector{T}, a::FixedBoundaryNetwork, ::Type{T}) where {T}
        for i in 1:a.Sx
            φ[i] = T(-6) + T(12) * T(i - 1) / T(max(a.Sx - 1, 1))
            φ[a.Sx + i] = T(-3) + T(6) * T(i - 1) / T(max(a.Sx - 1, 1))
        end
        return φ
    end

    """
        slab_bottom!(cache, data::FixedBoundaryData, problem, ansatz, sol)

    Fix the datum of the slab: at the first slab the level's parameters and the momentum of the
    initial condition, and at every later slab the previous level's solved parameters and the
    momentum it handed on; and the two ends' side data at the slab's start time, at its nodes and at
    its top time, with their derivatives in time.
    """
    function slab_bottom!(c::SpaceTimeGalerkinCache{ST}, data::FixedBoundaryData{ST},
            problem::LagrangianPDEProblem, ansatz::FixedBoundaryNetwork, sol) where {ST}
        a = ansatz
        t₀ = convert(ST, slab_start(problem, sol))
        h = convert(ST, timestep(problem))
        data.start = t₀
        if c.steps == 0
            side_row!(data, problem, 1, t₀)
            first_datum!(data, problem, a, t₀)
        else
            copyto!(data.phi, data.phi_next)
            copyto!(data.momentum, data.p_next)
            side_row!(data, problem, 1, t₀)
        end
        for q in eachindex(a.tabulation.tnode)
            side_row!(data, problem, q + 1, t₀ + h * convert(ST, a.tabulation.tnode[q]))
        end
        side_row!(data, problem, length(a.tabulation.tnode) + 2, t₀ + h)
        return nothing
    end

    # ---- the predictor -------------------------------------------------------------------

    """The uniform grid of [`PREDICTOR_POINTS`](@ref) points the predictor runs on, from the span's
    two ends."""
    function predictor_grid(a::FixedBoundaryNetwork, ::Type{T}) where {T}
        return collect(T, range(a.xspan[1], a.xspan[2]; length = PREDICTOR_POINTS))
    end

    """The value of one row of the predictor's history at a point `x` of the span, by linear
    interpolation on the uniform grid."""
    function predictor_interpolate(
            history::AbstractArray{ST, 3}, row::Integer, x, d::Integer,
            grid) where {ST}
        j = clamp(floor(Int, (x - grid[1]) / (grid[2] - grid[1])) + 1, 1, length(grid) - 1)
        λ = (x - grid[j]) / (grid[j + 1] - grid[j])
        return (1 - λ) * history[row, j, d] + λ * history[row, j + 1, d]
    end

    """
        predictor_force!(acc, u, v, w, du, dw, density, params, dx, mass, D)

    The predictor's acceleration at the grid points, `(∂L/∂u − ∂ₓ(∂L/∂w)) / m`: the slope in space by
    central differences of the field, and the divergence of the density's flux by central differences
    of `∂L/∂w`. The two ends carry no acceleration, because the side data is imposed there at every
    substep.
    """
    function predictor_force!(acc::AbstractMatrix, u::AbstractMatrix, v::AbstractMatrix,
            w::AbstractMatrix, du::AbstractMatrix, dw::AbstractMatrix, density, params, dx,
            mass, D::Integer)
        n = size(u, 1)
        T = eltype(u)
        for r in 1:n
            if r == 1
                for d in 1:D
                    w[r, d] = (u[2, d] - u[1, d]) / dx
                end
            elseif r == n
                for d in 1:D
                    w[r, d] = (u[n, d] - u[n - 1, d]) / dx
                end
            else
                for d in 1:D
                    w[r, d] = (u[r + 1, d] - u[r - 1, d]) / (2 * dx)
                end
            end
        end
        for r in 1:n
            point_u = ntuple(d -> u[r, d], D)
            point_v = ntuple(d -> v[r, d], D)
            point_w = ntuple(d -> w[r, d], D)
            for d in 1:D
                du[r, d] = field_derivative_u(density, point_u, point_v, point_w, params, d)
                dw[r, d] = field_derivative_w(density, point_u, point_v, point_w, params, d)
            end
        end
        for d in 1:D
            acc[1, d] = zero(T)
            acc[n, d] = zero(T)
        end
        for r in 2:(n - 1), d in 1:D

            acc[r, d] = (du[r, d] - (dw[r + 1, d] - dw[r - 1, d]) / (2 * dx)) / mass[d]
        end
        return acc
    end

    """
        predictor_sample!(sample, slab, cache, problem, sol)

    Run the predictor of one slab — Störmer–Verlet with [`PREDICTOR_SUBSTEPS`](@ref) substeps over
    [`PREDICTOR_POINTS`](@ref) points of the span, with the side data imposed at every substep — and
    fill `sample` at the nodes of the ansatz's rule by linear interpolation in `x` and in `t̂`, and the
    block's `top_sample` at `t̂ = 1` on the space nodes.

    The run reads the slab's datum from the coupling's block — the bottom level's parameters, the two
    ends' side data and the slab's start time — and not from the block's own copy of it, which the first
    residual evaluation writes after the guess that runs this predictor. It starts from the bottom
    level's network and the initial data's velocity at the first slab, and from the previous slab's top
    velocity after it: the coupling keeps that velocity when it writes the solved slab, at the grid
    this predictor runs on, so that the guess reads a state no later step has rewritten.
    """
    function predictor_sample!(sample::AbstractArray{ST, 3}, s::FixedBoundarySlab{ST},
            c::SpaceTimeGalerkinCache{ST, AD, FixedBoundaryData{ST}},
            problem::LagrangianPDEProblem, sol) where {ST, AD}
        data = c.coupling
        a = s.ansatz
        D, tb = a.D, a.tabulation
        NG, K = PREDICTOR_POINTS, PREDICTOR_SUBSTEPS
        grid = predictor_grid(a, ST)
        dx = grid[2] - grid[1]
        h = convert(ST, timestep(problem))
        dt = h / K
        params = parameters(problem)
        density = problem.density
        mass = predictor_mass(density, params, ST, D)
        φ0 = data.phi
        u = zeros(ST, NG, D)
        v = zeros(ST, NG, D)
        for r in 1:NG, d in 1:D

            u[r, d] = spatial_network(a, φ0, grid[r], data.ga[1, d], data.gb[1, d], d)
        end
        if c.steps == 0
            initial = problem.ics_function(grid, params)
            for r in 1:NG, d in 1:D

                v[r, d] = ST(initial.v[r, d])
            end
        else
            copyto!(v, data.v_top)
        end
        acc = zeros(ST, NG, D)
        w = zeros(ST, NG, D)
        du = zeros(ST, NG, D)
        dw = zeros(ST, NG, D)
        history = zeros(ST, K + 1, NG, D)
        copyto!(view(history, 1, :, :), u)
        predictor_force!(acc, u, v, w, du, dw, density, params, dx, mass, D)
        for k in 1:K
            for r in 1:NG, d in 1:D

                v[r, d] += dt / 2 * acc[r, d]
                u[r, d] += dt * v[r, d]
            end
            bcs = problem.bcs_function(data.start + k * dt, params)
            for d in 1:D
                u[1, d] = ST(bcs.left.u[d])
                u[NG, d] = ST(bcs.right.u[d])
            end
            predictor_force!(acc, u, v, w, du, dw, density, params, dx, mass, D)
            for r in 1:NG, d in 1:D

                v[r, d] += dt / 2 * acc[r, d]
                history[k + 1, r, d] = u[r, d]
            end
        end
        for q in axes(sample, 1)
            position = tb.tnode[q] * K
            k = clamp(floor(Int, position) + 1, 1, K)
            λ = position - (k - 1)
            for r in axes(sample, 2), d in axes(sample, 3)

                below = predictor_interpolate(history, k, tb.xnode[r], d, grid)
                above = predictor_interpolate(history, k + 1, tb.xnode[r], d, grid)
                sample[q, r, d] = (1 - λ) * below + λ * above
            end
        end
        for r in axes(s.top_sample, 1), d in axes(s.top_sample, 2)

            s.top_sample[r, d] = predictor_interpolate(history, K + 1, tb.xnode[r], d, grid)
        end
        return nothing
    end

    """
        evaluate!(sample, ::Predictor, cache, ansatz, problem, sol)

    A `Predictor` is a target of a [`FixedBoundaryNetwork`](@ref) alone: another ansatz has no slab
    predictor to run, and the construction raises `ArgumentError` rather than fitting a target
    nothing filled.
    """
    function evaluate!(sample::AbstractArray{ST, 3}, ::Predictor,
            c::SpaceTimeGalerkinCache, ansatz::SpaceTimeAnsatz,
            problem::LagrangianPDEProblem, sol) where {ST}
        return refuse_predictor(ansatz)
    end

    # ---- the fit that starts a slab ------------------------------------------------------

    """
        fit!(w, method::BubbleGreedy, ansatz::FixedBoundaryNetwork, sample, cache)

    Fit the whole parameter vector of a slab to the sampled target: the level's parameters by
    Levenberg–Marquardt from the datum to the target at `t̂ = 1` on the space nodes, and then the
    bubble, by the greedy selection of [`bubble_atoms`](@ref), to the target's remainder beyond the
    interpolation of the two levels and the lift.

    The workspace of the fit is the block's, rebuilt when the sizes change or when the method's
    parameters differ from the ones that built it.
    """
    function fit!(w::AbstractVector{ST}, method::BubbleGreedy,
            ansatz::FixedBoundaryNetwork, sample::AbstractArray{ST, 3},
            c::SpaceTimeGalerkinCache) where {ST}
        fit_spatial!(w, method, ansatz, c.coupling, c.ansatz.top_sample)
        fit_bubble!(w, method, ansatz, c.coupling, sample, c.ansatz)
        return w
    end

    function fit!(w::AbstractVector, method::SlabFitMethod, ansatz::FixedBoundaryNetwork,
            sample, c::SpaceTimeGalerkinCache)
        throw(ArgumentError(
            "a FixedBoundaryNetwork is fitted by BubbleGreedy, got $(nameof(typeof(method)))"))
    end

    # The level: Levenberg–Marquardt from the datum's parameters to the target at the top of the
    # slab, on the space nodes of the rule weighted by the square root of its weights.
    function fit_spatial!(w::AbstractVector{ST}, method::BubbleGreedy,
            a::FixedBoundaryNetwork, data::FixedBoundaryData{ST},
            top::AbstractMatrix{ST}) where {ST}
        nodes, weights, D = a.tabulation.xnode, a.tabulation.xweight, a.D
        ga1 = [data.ga[end, d] for d in 1:D]
        gb1 = [data.gb[end, d] for d in 1:D]
        r!(b, φ) = begin
            k = 0
            for r in eachindex(nodes), d in 1:D

                k += 1
                b[k] = sqrt(weights[r]) *
                       (spatial_network(a, φ, nodes[r], ga1[d], gb1[d], d) - top[r, d])
            end
            b
        end
        φ = collect(data.phi)
        least_squares!(φ, r!, length(nodes) * D; iterations = method.iterations)
        copyto!(w, 1, φ, 1, nspatial(a))
        return w
    end

    # The bubble: the target's remainder beyond the two levels and the lift, and the greedy selection
    # of the atoms whose correlation with it is largest.
    function fit_bubble!(w::AbstractVector{ST}, method::BubbleGreedy,
            a::FixedBoundaryNetwork, data::FixedBoundaryData{ST},
            sample::AbstractArray{ST, 3}, s::FixedBoundarySlab{ST}) where {ST}
        tb = a.tabulation
        qt, qx, D, St = length(tb.tnode), length(tb.xnode), a.D, a.St
        N = qt * qx
        atoms = bubble_atoms(method, ST)
        f = s.fit
        if f.method != method || size(f.table) != (size(atoms, 1), N) ||
           size(f.residual, 2) != D || size(f.chosen, 2) != St ||
           size(f.atoms) != size(atoms)
            f.atoms = atoms
            f.table = zeros(ST, size(atoms, 1), N)
            f.target = zeros(ST, N, D)
            f.residual = zeros(ST, N, D)
            f.correlations = zeros(ST, size(atoms, 1), D)
            f.coefficients = zeros(ST, St, D)
            f.selected = zeros(Int, St)
            f.taken = falses(size(atoms, 1))
            f.weighted = zeros(ST, N)
            f.chosen = zeros(ST, N, St)
            f.gram = zeros(ST, St, St)
            for q in 1:qt, r in 1:qx

                k = (q - 1) * qx + r
                t̂ = tb.tnode[q]
                ŝ = (tb.xnode[r] - a.xspan[1]) / (a.xspan[2] - a.xspan[1])
                bump = (tb.xnode[r] - a.xspan[1]) * (a.xspan[2] - tb.xnode[r])
                ω = sqrt(tb.tweight[q] * tb.xweight[r]) * t̂ * (1 - t̂) * bump
                f.weighted[k] = sqrt(tb.tweight[q] * tb.xweight[r])
                for i in axes(f.atoms, 1)
                    f.table[i, k] = ω * tanh(f.atoms[i, 1] * t̂ + f.atoms[i, 2] * ŝ +
                                         f.atoms[i, 3])
                end
            end
        end
        f.method = method
        # the remainder of the target beyond the two levels and the lift, at the slab's nodes: the
        # solved level is the one this fit is building, not the block's buffer, which the guess's
        # own fit has already overwritten
        φ0, φ1 = data.phi, view(w, 1:nspatial(a))
        for q in 1:qt, r in 1:qx

            k = (q - 1) * qx + r
            t̂ = tb.tnode[q]
            x = tb.xnode[r]
            ŝ = (x - a.xspan[1]) / (a.xspan[2] - a.xspan[1])
            for d in 1:D
                ga0, gb0 = data.ga[1, d], data.gb[1, d]
                ga1, gb1 = data.ga[end, d], data.gb[end, d]
                n0 = spatial_network(a, φ0, x, ga0, gb0, d)
                n1 = spatial_network(a, φ1, x, ga1, gb1, d)
                ra = data.ga[q + 1, d] - (1 - t̂) * ga0 - t̂ * ga1
                rb = data.gb[q + 1, d] - (1 - t̂) * gb0 - t̂ * gb1
                base = (1 - t̂) * n0 + t̂ * n1 + (1 - ŝ) * ra + ŝ * rb
                f.target[k, d] = f.weighted[k] * (sample[q, r, d] - base)
            end
        end
        copyto!(f.residual, f.target)
        fill!(f.taken, false)
        taken = 0
        while taken < St
            mul!(f.correlations, f.table, f.residual)
            best, pick = zero(ST), 0
            for i in axes(f.table, 1), d in 1:D

                f.taken[i] && continue
                if abs(f.correlations[i, d]) > best
                    pick, best = i, abs(f.correlations[i, d])
                end
            end
            iszero(best) && break
            f.taken[pick] = true
            taken += 1
            f.selected[taken] = pick
            A = transpose(f.table[f.selected[1:taken], :])
            f.coefficients[1:taken, :] .= A \ f.target
            f.residual .= f.target .- A * view(f.coefficients, 1:taken, :)
        end
        fill!(view(w, (nspatial(a) + 1):length(w)), zero(ST))
        for j in 1:taken
            atom = view(f.atoms, f.selected[j], :)
            if a.bubble_form === :tsr
                θ, s_, ρ = bubble_chart(atom[1], atom[2], atom[3], j)
                w[nspatial(a) + j] = θ
                w[nspatial(a) + St + j] = s_
                w[nspatial(a) + 2 * St + j] = ρ
            else
                w[nspatial(a) + j] = atom[1]
                w[nspatial(a) + St + j] = atom[2]
                w[nspatial(a) + 2 * St + j] = atom[3]
            end
            for d in 1:D
                w[nspatial(a) + 3 * St + (j - 1) * D + d] = f.coefficients[j, d]
            end
        end
        return w
    end
end
