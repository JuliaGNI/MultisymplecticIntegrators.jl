module MultisymplecticIntegratorsPlots

using Makie

# `using Makie` exports a wide set of names of this ecosystem — its `TimeSeries` recipe is the one
# that clashes with `GeometricSolutions`' — so the names of this package are reached through the
# module (`M.solution_data`, `M.LagrangianPDEProblem`), and the names used bare here are Makie's.
import MultisymplecticIntegrators
import MultisymplecticIntegrators: Diagnostics
import Makie: wong_colors

const M = MultisymplecticIntegrators

# ---- the one theme -----------------------------------------------------------
#
# Every font size, font weight, line width, marker size and colour of every figure here comes from
# this theme: no method below sets one of its own. That is what makes a figure of this package sit
# beside one of the ecosystem's other packages without a visible change of typeface size, and it is
# why the methods apply the theme themselves: the attributes named here are this theme's in every
# figure, whatever ambient theme the caller set.
#
# The colour cycle is Wong's palette as `wong_colors()` provides it, which is colour-blind safe;
# its order is fixed, so a role keeps its colour across a set of figures.
const PLOT_THEME = Theme(
    fontsize = 18,
    palette = (color = wong_colors(),),
    Label = (font = :bold,),
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

# Four times as wide as tall per panel, so that figures of different panel counts can be included at
# one width; a caller passes `figsize` for anything else.
const FIGURE_WIDTH = 1200
const PANEL_HEIGHT = 300

# ---- the data of a figure ----------------------------------------------------
#
# A figure is drawn from plain vectors: the step times, the space grid, and one `nt × nx` matrix of
# the field per component. The run of `integrate` and the archive `scripts/archives.jl` writes both
# reduce to that, which is what keeps the drawing below free of problems and solution objects.

function solution_fields(problem::M.LagrangianPDEProblem, result)
    t, x, u = M.solution_data(problem, result)
    return (t = t, x = x, u = u, heading = "")
end

function solution_fields(data::AbstractDict)
    t = collect(data["t"])
    x = collect(data["x"])
    T = promote_type(eltype(t), eltype(x))
    u = [convert(Matrix{T}, component) for component in data["u"]]
    return (t = convert(Vector{T}, t), x = convert(Vector{T}, x), u = u,
        heading = _solution_heading(data))
end

# A heading naming the run, for a figure taken out of the directory it was drawn in.
function _solution_heading(data)
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

function Diagnostics.plot_solution(problem::M.LagrangianPDEProblem, result; path = nothing,
        title = nothing, figsize = nothing)
    return draw_solution(solution_fields(problem, result); path = path, title = title,
        figsize = figsize)
end

function Diagnostics.plot_solution(data::AbstractDict; path = nothing, title = nothing,
        figsize = nothing)
    return draw_solution(solution_fields(data); path = path, title = title, figsize = figsize)
end

function draw_solution(fields; path = nothing, title = nothing, figsize = nothing)
    heading = title === nothing ? fields.heading : title
    figure = with_theme(PLOT_THEME) do
        D = length(fields.u)
        fig = Figure(size = _figsize(figsize, D))
        isempty(heading) ||
            Label(fig[0, 1], heading; tellwidth = false)
        for d in 1:D
            ax = Axis(fig[d, 1]; xlabel = "x", ylabel = "t",
                title = D == 1 ? "" : "component $d")
            # An interval range and not the grid: a heatmap of a matrix takes the extent of the
            # cells, which is what a vector format such as PDF can place. Makie lays the first
            # dimension of the matrix along the first interval, so the `nt × nx` field is
            # transposed to put space across and time up.
            shading = heatmap!(
                ax, fields.x[1] .. fields.x[end], fields.t[1] .. fields.t[end],
                permutedims(fields.u[d]))
            Colorbar(fig[d, 2], shading; label = "u")
        end
        fig
    end
    path === nothing || save(path, figure)
    return figure
end

# ---- the energy error over time ---------------------------------------------

function Diagnostics.plot_energy_error(times, errors; labels = nothing, path = nothing,
        title = nothing, figsize = nothing)
    length(errors) == length(labels === nothing ? errors : labels) ||
        throw(ArgumentError("$(length(errors)) error series against $(length(labels)) labels"))
    heading = title === nothing ? "" : title
    figure = with_theme(PLOT_THEME) do
        fig = Figure(size = _figsize(figsize, 1))
        isempty(heading) || Label(fig[0, 1], heading; tellwidth = false)
        ax = Axis(fig[1, 1]; xlabel = "t", ylabel = "|ΔH / H₀|", yscale = log10)
        drawn = 0
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
            drawn += 1
        end
        # A legend names the series the axis carries, so it is drawn only when there is one: over an
        # empty axis Makie refuses it, and a figure of nothing is a failed draw rather than a
        # comparison.
        iszero(drawn) || Legend(fig[1, 2], ax; framevisible = false)
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

# ---- figures from an archive --------------------------------------------------
#
# The layer above the three figures: given one archive, draw the figure its kind asks for. The
# archive is a plain dictionary of flat keys — see the schema in `scripts/README.md` — so this knows
# nothing about how a run was stored or which study made it, and nothing about where the figure is
# saved.

function Diagnostics.figures(data::AbstractDict)
    haskey(data, "kind") ||
        throw(ArgumentError("the archive has no \"kind\"; cannot tell what to draw"))
    kind = data["kind"]
    if kind == "solution"
        return Diagnostics.plot_solution(data)
    elseif kind == "energy"
        return Diagnostics.plot_energy_error(data)
    end
    throw(ArgumentError("the archive has kind \"$(kind)\", and this module draws \"solution\" " *
                        "and \"energy\""))
end

end
