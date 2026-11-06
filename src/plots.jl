"""
    MultisymplecticIntegrators.Diagnostics

Figures for the integrators of this package: a space-time plot of a solution, the energy error of
several runs over time, an animation of the field, and the dispatcher that turns one archived run
into the figure its kind asks for.

The methods are implemented in the `MultisymplecticIntegratorsPlots` extension, which loads
together with `Makie` or a backend such as `CairoMakie`. The declarations here hold the API and its
documentation, so that each name is documented whether or not a backend is loaded, and the
extension documents no name a second time.

# Why a submodule rather than three more exports

`plot_solution` and `plot_energy_error` are names `GeometricProblems` already exports —
`plot_solution` from its problem submodules and `plot_energy_error` from its own `Diagnostics`.
Exporting them from this package's top level would make them ambiguous in any scope that also does
`using GeometricProblems.Wave`-style imports, which is what a script integrating a problem naturally
writes. The extension therefore defines `Diagnostics.plot_solution`, which extends the declaration
rather than shadowing a name, and a caller reaches it as
`MultisymplecticIntegrators.Diagnostics.plot_solution`.
"""
module Diagnostics

export plot_theme, plot_solution, plot_energy_error, animate_solution, figures

"""
    plot_theme()

The one theme of every figure of this package: the font sizes and weight, the line and marker
sizes, and the colour cycle.

A function rather than a value, because a Makie `Theme` can only exist where Makie does, and this
module is loaded whether or not a backend is. The extension defines it, and applies it inside each
figure method, so that no method sets a font, a line width or a colour of its own: the attributes
the theme names are the theme's in every figure, whatever ambient theme the caller set.
"""
function plot_theme end

"""
    plot_solution(problem, result; path = nothing, title = nothing, figsize = nothing)
    plot_solution(data::AbstractDict; kwargs...)

A space-time plot of one run: one heatmap panel per component of the field, the time axis running
up the page and the problem's output grid across it, with the values read off the panel's
colourbar.

`result` is what `integrate` returns, and `problem` the problem it was integrated with. The
dictionary form reads the archive that `scripts/archives.jl` writes, whose `"kind"` is `"solution"`.

# Keyword arguments

  - `path = nothing`: write the figure there, as a PDF by its extension, beside returning it.
  - `title = nothing`: the heading across the figure. `nothing` draws the heading of the archive,
    which names the run and its time step, and the form without an archive draws none; an empty
    string draws none either, whatever the archive holds.
  - `figsize = nothing`: the size of the figure, in pixels, as `(width, height)`.
"""
function plot_solution end

"""
    plot_energy_error(times, errors; labels = nothing, path = nothing, title = nothing, figsize = nothing)
    plot_energy_error(data::AbstractDict; kwargs...)

The relative energy error of several runs against time, on a logarithmic axis.

`errors` holds one series per run, in
[`energy_error`](@ref MultisymplecticIntegrators.energy_error)'s units, and `times` either one time
vector shared by every series or one per series. A zero error cannot be drawn on a logarithmic axis,
so its point is dropped from both vectors rather than masked, which keeps the curve whole. A series
with no drawable point is left out of the figure, and the legend names the series that were drawn:
a legend over an empty axis claims a result that the figure does not show.

The dictionary form reads an archive whose `"kind"` is `"energy"`.

# Keyword arguments

  - `labels = nothing`: one legend entry per series.
  - `path = nothing`: write the figure there, as a PDF by its extension.
  - `title = nothing`, `figsize = nothing`: as for [`plot_solution`](@ref); the form without an
    archive draws no heading.
"""
function plot_energy_error end

"""
    animate_solution(problem, result; path, framerate = 15, title = "", figsize = nothing)

An animation of the field `u(t, ·)` over the stored steps of one run, written to `path`.

Only the first component of the field is animated, whatever the problem's dimension `D`.

`result` is what `integrate` returns and `problem` the problem it was integrated with. The frame at
step `n` is the field on the problem's output grid, at the time of that step, so the animation shows
the solution of the discrete problem and nothing interpolated between the steps.
"""
function animate_solution end

"""
    figures(data) -> Figure

The figure one archive earns: a `"solution"` archive draws as a space-time plot, a `D`-component
solution as one panel per component, and an `"energy"` archive as the energy error of its series.

`data` is a run's archive as a plain dictionary — the schema is documented in
`scripts/README.md`. Its `"kind"` selects the shape of figure, and a kind this module cannot draw
is refused rather than guessed at: two figures drawn from the wrong keys look like a result.

This is what makes a renderer a dispatcher — glob a run directory, hand each archive over, save
what comes back — rather than a second registry of which experiment produces which picture, kept in
step with the first by hand. Naming the file is the caller's, from the archive's `"stem"`: the
figure does not know where output goes.
"""
function figures end

end
