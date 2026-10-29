"""
    MultisymplecticIntegrators.Diagnostics

Figures for the integrators of this package: a space-time plot of a solution, the energy error of
several runs over time, an animation of the field, and the dispatcher that turns one archived run
into the figure its kind asks for.

The methods are implemented in the `MultisymplecticIntegratorsPlots` extension, which loads
together with `Makie` or a backend such as `CairoMakie`. The declarations here hold the API, so that
each name is documented whether or not a backend is loaded.

# Why a submodule rather than three more exports

`plot_solution` and `plot_energy_error` are names `GeometricProblems` already exports, from every
problem submodule and from its own `Diagnostics`. Exporting them from this package's top level would
make them ambiguous in any scope that also does `using GeometricProblems.Wave`-style imports, which
is what a script integrating a problem naturally writes. The extension therefore defines
`Diagnostics.plot_solution`, which extends the declaration rather than shadowing a name, and a
caller reaches it as `MultisymplecticIntegrators.Diagnostics.plot_solution`.
"""
module Diagnostics

export plot_theme, plot_solution, plot_energy_error, animate_solution, figures

"""
    plot_theme()

The one theme of every figure of this package: the font sizes and weight, the line and marker
sizes, and the colour cycle.

A function rather than a value, because a Makie `Theme` can only exist where Makie does, and this
module is loaded whether or not a backend is. The extension defines it, and applies it inside each
figure method, so that no method sets a size, a font or a colour of its own: the attributes the
theme names are the theme's in every figure, whatever ambient theme the caller set.
"""
function plot_theme end

"""
    plot_solution(problem, result; path = nothing, title = "", figsize = nothing)
    plot_solution(data::AbstractDict; kwargs...)

A space-time plot of one run: one heatmap panel per component of the field, the time axis running
down the page and the problem's output grid across it, with the values read off the panel's
colourbar.

`result` is what `integrate` returns, and `problem` the problem it was integrated with. The
dictionary form reads the archive that `scripts/archives.jl` writes, whose `"kind"` is `"solution"`.

# Keyword arguments

  - `path = nothing`: write the figure there, as a PDF by its extension, beside returning it.
  - `title = ""`: a heading across the figure; the form without one takes the archive's.
  - `figsize = nothing`: the size of the figure, in pixels, as `(width, height)`.
"""
function plot_solution end

"""
    plot_energy_error(times, errors; labels = nothing, path = nothing, title = "", figsize = nothing)
    plot_energy_error(data::AbstractDict; kwargs...)

The relative energy error of several runs against time, on a logarithmic axis.

`errors` holds one series per run, in
[`energy_error`](@ref MultisymplecticIntegrators.energy_error)'s units, and `times` either one time
vector shared by every series or one per series. A zero error cannot be drawn on a logarithmic axis,
so its point is dropped from both vectors rather than masked, which keeps the curve whole.

The dictionary form reads an archive whose `"kind"` is `"energy"`.

# Keyword arguments

  - `labels = nothing`: one legend entry per series.
  - `path = nothing`: write the figure there, as a PDF by its extension.
  - `title = ""`, `figsize = nothing`: as for [`plot_solution`](@ref).
"""
function plot_energy_error end

"""
    animate_solution(problem, result; path, framerate = 15, title = "", figsize = nothing)

An animation of the field `u(t, ·)` over the stored steps of one run, written to `path`.

`result` is what `integrate` returns and `problem` the problem it was integrated with. The frame at
step `n` is the field on the problem's output grid, at the time of that step, so the animation shows
the solution of the discrete problem and nothing interpolated between the steps.
"""
function animate_solution end

"""
    figures(data) -> Vector{Pair{String, Figure}}

Every figure one archive earns, each paired with the name it belongs under.

`data` is a run's archive as a plain dictionary — the schema is documented in
`scripts/archives.jl`. Its `"kind"` selects the shape of figure and its `"stem"` names it, and a
kind this module cannot draw is refused rather than guessed at: two figures drawn from the wrong
keys look like a result.

This is what makes a renderer a dispatcher — glob a run directory, hand each archive over, save what
comes back — rather than a second registry of which experiment produces which picture, kept in step
with the first by hand. The caller does the saving, because where output goes is a property of the
caller and not of the figure.
"""
function figures end

end
