# Scripts

Exploratory drivers and studies. Nothing here is run by the docs build or by CI — these are for
investigating one question at a time.

**Code here, data in `runs/`, figures in `results/`.** Both output directories are at the repository
root and git-ignored. The driver and the renderer, and the scans that archive, take `--runs-dir`
and `--results-dir`, so the same run can write into a paper's or a talk's figure directory without
being copied there.

```sh
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
```

`scripts/` is a project of its own. A script started from the repository root would load the
package's environment and find neither `JLD2` nor `CairoMakie`. Every `[deps]` entry of
`scripts/Project.toml` carries a `[compat]` entry, as the package's does.

## The harness

One registry, one driver, one model file per family, one renderer. Each stage reads what the
previous one wrote.

```sh
julia --project=scripts scripts/run_family.jl spline     # the smallest spline entry → runs/<stem>.jld2
julia --project=scripts scripts/figures.jl               # every archive → the PDFs in results/
```

The driver takes the family name first and then the stems of the entries to run, so one figure can
be regenerated on its own:

```sh
julia --project=scripts scripts/run_family.jl spline wave-spline-h0.2 wave-spline-h0.1
julia --project=scripts scripts/figures.jl wave-spline-h0.1
julia --project=scripts scripts/spline_energy_scan.jl --steps 0.2,0.1
julia --project=scripts scripts/spline_phase_drift_scan.jl
julia --project=scripts scripts/profile_residual.jl --steps 0.05,0.025
```

With no stem named, the driver runs the cheapest entry of the family: the largest time step, which
is what answers "does this family run at all" in the least time. A run that throws is reported on
stderr with its backtrace, archives nothing, and makes the driver exit 1, so a failed family is
distinguishable from a family that ran.

**The solves and the figures are separate stages, with `runs/` between them.** A figure can be
restyled without paying for the solves again, and a missing figure stays distinguishable from a
failed run.

| file | does |
|:--|:--|
| `archives.jl` | where output goes, `atomic_write`, the archive's writer and its reader, the option parser, the printing helpers. Depends on `JLD2` and `Printf` alone, so a renderer loads it cheaply. No work at top level. |
| `experiments.jl` | the registry: `register_family!`, `family_entries`, `family_run`, and the two selection rules `smallest_entry` and `entry_by_stem`, plus the ladders of the two scans. It holds no family's entries and builds nothing on `include`. |
| `spline_model.jl` | the spline family: its registry entries, the problem and the method of one of them, one run timed, the archive of a run, and the error of a run against the exact solution at its last stored step. It ends by registering `:spline`. |
| `run_family.jl` | the one driver: it includes every `*_model.jl` file here in sorted order, takes the family name and the stems of its entries, archives each run under `runs/`, and exits non-zero if one failed. |
| `figures.jl` | globs `runs/`, draws each archive as the kind it declares, and writes `results/<stem>.pdf` through `atomic_write`. **The only place `save` is called.** |
| `spline_energy_scan.jl` | the relative energy error over time at a ladder of time steps, archived as one `"energy"` run that `figures.jl` draws as a comparison. |
| `spline_phase_drift_scan.jl` | the error against the exact solution at the final time, at a ladder of ansatz resolutions. Prints a table and writes nothing. |
| `profile_residual.jl` | one slab's residual, timed, with its allocations and the cost of a full step. |
| `weak_initial_condition_reference.jl` | the reference run of `WeakInitialCondition` on `Wave`, written to `test/helpers/weak_initial_condition_reference.jl` with the commit, the Julia version, the `--check-bounds` flag and the dependency versions in its header. Runs in the package's own environment: `julia --startup-file=no --project=. --check-bounds=yes scripts/weak_initial_condition_reference.jl`. |
| `linear_solver_accuracy.jl` | the direct linear solve (`linear_solver = true`) against Newton, slab by slab, on the runs of the `linear_solver` testset of `test/ansatz/tensor_bspline.jl`, in `Float32` and `Float64`, and both against a `Float64` Newton reference. Prints a table and writes nothing. Runs in the package's own environment: `julia --startup-file=no --project=. scripts/linear_solver_accuracy.jl`. |
| `action_residual_spike.jl` | the slab residual and Jacobian as the AD gradient and Hessian of the property suite's discrete action (`test/helpers/action_oracle.jl`), checked against `residual!` and the solver's Jacobian and then timed against them, with ForwardDiff and with Enzyme (Mooncake where Enzyme fails), one fresh process per backend and problem. Prints a table and writes nothing. |

The spike's gate is `--quick`, the smallest size with one sample, and `--selfcheck`, which must exit
non-zero because both of its checks reject a perturbed vector:

```sh
julia --startup-file=no --project=scripts scripts/action_residual_spike.jl --quick
julia --startup-file=no --project=scripts scripts/action_residual_spike.jl --selfcheck
julia --startup-file=no --project=scripts scripts/action_residual_spike.jl --sizes 60,120,250
```

## Adding a family

A family is one file beside `run_family.jl`, named `<family>_model.jl`, that builds its entries and
the function which runs one of them, and registers both:

```julia
const NETWORK = ((stem = "wave-network-h0.2", label = "network, h = 0.2", …),)

function network_run(entry)
    …
    return Dict{String, Any}("kind" => "solution", "label" => entry.label, …)
end

register_family!(:network, NETWORK; run = network_run)
```

An entry is a named tuple with at least `stem`, which names its archive and its figure, and `label`,
which a report and a heading show; `smallest_entry` reads `timestep` and `xstep` where an entry
carries them. `run(entry)` returns the archive as a `Dict{String, Any}` with a `"kind"`, and
`store_run!` adds the `"stem"` and writes the file. Nothing else needs an edit: the driver includes
every model file, and the registry refuses a second family of the same name.

## No figure code lives here

Figures are built by `MultisymplecticIntegrators.Diagnostics`, in the `MultisymplecticIntegratorsPlots`
extension, which `using CairoMakie` activates. `figures.jl` reads an archive, hands it to the
diagnostics, and saves what comes back:

| want | use |
|:--|:--|
| the field over the whole slab | `Diagnostics.plot_solution` |
| the energy error of several runs over time | `Diagnostics.plot_energy_error` |
| an animation of the field over the stored steps | `Diagnostics.animate_solution` |
| the figure one archive earns, by its `"kind"` | `Diagnostics.figures` |

`Diagnostics.figures` returns the figure and names nothing: the archive's `"stem"` is what
`figures.jl` calls the file, so the naming is the renderer's and the drawing is the extension's.

Fonts, line widths and colours come from `Diagnostics.plot_theme()`, which the figure methods
apply themselves, so changing the theme restyles every figure. A figure's own extent is not the
theme's: the extension sizes it from `FIGURE_WIDTH` and `PANEL_HEIGHT` per panel, and the `figsize`
keyword overrides that.

## The archive schema

One JLD2 file per run, `runs/<stem>.jld2`. `"kind"` says what shape of figure the run draws and
`"stem"` names the archive and its figure; `store_run!` refuses a data set without a `"kind"`, and
`figures.jl` refuses a kind it cannot draw — a figure guessed at would be drawn from the wrong keys.

| kind | keys |
|:--|:--|
| `"solution"` | `"kind"`, `"stem"`, `"label"`, `"problem_label"`, `"timestep"`, `"t"`, `"x"`, `"u"` (one matrix per component) |
| `"energy"` | `"kind"`, `"stem"`, `"title"`, `"labels"`, `"t"` and `"errors"` (one vector per series) |

Every key is read: the `"solution"` figure reads the run's identity, its time step and its three
value arrays, and nothing else is stored. A run's energy error is reported by the run that computed
it rather than archived, because it is a diagnostic of the run and a stored copy could only disagree
with it.

Every file the scripts write under `--runs-dir` and `--results-dir` is written through
`atomic_write`: a sibling temporary carrying the destination's extension, renamed over it. Its name
starts with a dot, which the archive reader skips, so a temporary that a killed writer leaves behind
is never read as a result. A writer that throws leaves the earlier file whole and removes its
temporary, so a failed write is never read as a result either.

### Naming

**One scheme: `<problem>-<family>-h<timestep>`**, so sorting a directory groups the methods of one
problem together and the steps of one method; a run that compares several steps rather than fixing
one is named for its study instead (`wave-spline-energy`). An archive and its figure share the stem,
because the renderer globs the directory rather than consulting a registry.
