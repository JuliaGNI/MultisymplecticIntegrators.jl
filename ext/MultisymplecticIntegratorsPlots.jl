module MultisymplecticIntegratorsPlots

using Makie

# `using Makie` exports a wide set of names of this ecosystem — its `TimeSeries` recipe is the one
# that clashes with `GeometricSolutions`' — so nothing here takes a name from Makie's own export
# list unqualified, and the names of this package are reached through the module.
import MultisymplecticIntegrators
import MultisymplecticIntegrators: Diagnostics

const M = MultisymplecticIntegrators

# ---- the one theme -----------------------------------------------------------
#
# Every font size, line width, marker size and colour of every figure here comes from this theme:
# no method below sets one of its own. That is what makes a figure of this package sit beside one
# of the ecosystem's other packages without a visible change of typeface size, and it is why the
# methods apply the theme themselves — a caller who wants another look sets the ambient theme and
# is not overruled.
#
# The colour cycle is Makie's own, which is colour-blind safe and is what the rest of the
# ecosystem's Makie code uses; the order is fixed, so a role keeps its colour across a set.
const PLOT_THEME = Theme(
    fontsize = 18,
    palette = (color = Makie.wong_colors(),),
    Lines = (linewidth = 2,),
    Scatter = (markersize = 10,),
    Axis = (
        xlabelsize = 22,
        ylabelsize = 22,
        xticklabelsize = 16,
        yticklabelsize = 16,
        titlesize = 20
    )
)

Diagnostics.plot_theme() = PLOT_THEME

# Twice as wide as tall per panel, so that figures of different panel counts can be included at one
# width; a caller passes `figsize` for anything else.
const FIGURE_WIDTH = 1200
const PANEL_HEIGHT = 300

# ---- the data of a figure ----------------------------------------------------
#
# A figure is drawn from plain vectors: the step times, the space grid, and one `nt × nx` matrix of
# the field per component. The run of `integrate` and the archive `scripts/archives.jl` writes both
# reduce to that, which is what keeps the drawing below free of problems and solution objects.

struct SolutionFields{T}
    t::Vector{T}
    x::Vector{T}
    u::Vector{Matrix{T}}
    title::String
end

function solution_fields(problem::M.LagrangianPDEProblem, result)
    t, x, u = M.solution_data(problem, result)
    return SolutionFields{eltype(t)}(t, x, u, "")
end

function solution_fields(data::AbstractDict)
    t = collect(data["t"])
    x = collect(data["x"])
    T = promote_type(eltype(t), eltype(x))
    u = [convert(Matrix{T}, component) for component in data["u"]]
    title = get(data, "title", "")
    isempty(title) && (title = _solution_title(data))
    return SolutionFields{T}(convert(Vector{T}, t), convert(Vector{T}, x), u, title)
end

# A heading naming the run, for a figure taken out of the directory it was drawn in.
function _solution_title(data)
    name = get(data, "stem", "")
    label = get(data, "label", "")
    problem = get(data, "problem_label", "")
    parts = filter(!isempty, (problem, label))
    headline = isempty(parts) ? "" : join(parts, " — ")
    step = haskey(data, "timestep") ? ", Δt = $(data["timestep"])" : ""
    return isempty(headline) ? name : headline * step
end

function _figsize(figsize, panels)
    figsize === nothing ? (FIGURE_WIDTH, PANEL_HEIGHT * panels) : figsize
end

# ---- the space-time plot -----------------------------------------------------

"""
    plot_solution(problem, result; kwargs...)
    plot_solution(data::AbstractDict; kwargs...)

A space-time plot of one run: one heatmap panel per component of the field, the time axis running
down the page and the problem's output grid across it.

Implemented in the `MultisymplecticIntegratorsPlots` extension; the arguments and the keywords are
documented at the declaration in `src/plots.jl`.
"""
function Diagnostics.plot_solution(problem::M.LagrangianPDEProblem, result; path = nothing,
        title = nothing, figsize = nothing)
    return draw_solution(solution_fields(problem, result); path = path, title = title,
        figsize = figsize)
end

function Diagnostics.plot_solution(data::AbstractDict; path = nothing, title = nothing,
        figsize = nothing)
    return draw_solution(solution_fields(data); path = path, title = title, figsize = figsize)
end

function draw_solution(fields::SolutionFields; path = nothing, title = nothing,
        figsize = nothing)
    heading = title === nothing ? fields.title : title
    figure = with_theme(PLOT_THEME) do
        D = length(fields.u)
        fig = Figure(size = _figsize(figsize, D))
        isempty(heading) ||
            Label(fig[0, 1], heading; font = :bold, tellwidth = false)
        for d in 1:D
            ax = Axis(fig[d, 1]; xlabel = "x", ylabel = "t",
                title = D == 1 ? "" : "component $d")
            # An interval range and not the grid: a heatmap of a matrix takes the extent of the
            # cells, which is what a vector format such as PDF can place.
            heatmap = heatmap!(
                ax, fields.x[1] .. fields.x[end], fields.t[1] .. fields.t[end],
                fields.u[d])
            Colorbar(fig[d, 2], heatmap; label = "u")
        end
        fig
    end
    path === nothing || save(path, figure)
    return figure
end

# ---- the energy error over time ---------------------------------------------

"""
    plot_energy_error(times, errors; kwargs...)
    plot_energy_error(data::AbstractDict; kwargs...)

The relative energy error of several runs against time, on a logarithmic axis.

Implemented in the `MultisymplecticIntegratorsPlots` extension; the arguments and the keywords are
documented at the declaration in `src/plots.jl`.
"""
function Diagnostics.plot_energy_error(times, errors; labels = nothing, path = nothing,
        title = "", figsize = nothing)
    length(errors) == length(labels === nothing ? errors : labels) ||
        throw(ArgumentError("$(length(errors)) error series against $(length(labels)) labels"))
    figure = with_theme(PLOT_THEME) do
        fig = Figure(size = _figsize(figsize, 1))
        isempty(title) || Label(fig[0, 1], title; font = :bold, tellwidth = false)
        ax = Axis(fig[1, 1]; xlabel = "t", ylabel = "|ΔH / H₀|", yscale = log10)
        for (i, series) in enumerate(errors)
            t = collect(eltype(times) <: AbstractVector ? times[i] : times)
            e = collect(series)
            length(t) == length(e) || throw(ArgumentError(
                "series $(i) has $(length(e)) errors against $(length(t)) times"))
            # A logarithmic axis cannot carry a zero, and the first point of every series is one by
            # construction. The point is dropped from the time vector as well, so that the curve
            # stays whole instead of breaking at round-off.
            keep = [k for k in eachindex(e) if isfinite(e[k]) && e[k] > 0]
            isempty(keep) && continue
            lines!(ax, t[keep], e[keep];
                label = labels === nothing ? "run $i" : labels[i])
        end
        isempty(errors) || Legend(fig[1, 2], ax; framevisible = false)
        fig
    end
    path === nothing || save(path, figure)
    return figure
end

function Diagnostics.plot_energy_error(data::AbstractDict; path = nothing, title = nothing,
        figsize = nothing)
    times = data["t"]
    errors = data["errors"]
    labels = get(data, "labels", nothing)
    heading = title === nothing ? get(data, "title", "") : title
    return Diagnostics.plot_energy_error(times, errors; labels = labels, path = path,
        title = heading, figsize = figsize)
end

# ---- the animation -----------------------------------------------------------

"""
    animate_solution(problem, result; path, kwargs...)

An animation of the field `u(t, ·)` over the stored steps of one run.

Implemented in the `MultisymplecticIntegratorsPlots` extension; the arguments and the keywords are
documented at the declaration in `src/plots.jl`.
"""
function Diagnostics.animate_solution(problem::M.LagrangianPDEProblem, result; path,
        framerate = 15, title = "", figsize = nothing)
    fields = solution_fields(problem, result)
    figure = with_theme(PLOT_THEME) do
        u = fields.u[1]
        fig = Figure(size = _figsize(figsize, 1))
        ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "u",
            title = isempty(title) ? "" : title)
        profile = Observable(collect(view(u, 1, :)))
        lines!(ax, fields.x, profile)
        record(fig, path, axes(u, 1); framerate = framerate) do n
            profile[] = collect(view(u, n, :))
            ax.title = isempty(title) ? "t = $(fields.t[n])" :
                       "$(title), t = $(fields.t[n])"
            return nothing
        end
        fig
    end
    return figure
end

end
