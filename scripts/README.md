# Scripts

Exploratory drivers and studies. Nothing here is run by the docs build or by CI — these are for
investigating one question at a time.

**Code here, data in `runs/`, figures in `results/`.** Both output directories are at the repository
root and git-ignored. Every driver takes `--runs-dir` and `--results-dir`, so the same script can
write into a paper's or a talk's figure directory without being copied there.

```sh
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
```

`scripts/` is a project of its own. A driver started from the repository root would load the
package's environment and find neither `JLD2` nor `CairoMakie`.

## The harness

One registry, three drivers, one renderer. Each stage reads what the previous one wrote.

```sh
julia --project=scripts scripts/run_spline.jl        # the smallest spline entry → runs/<stem>.jld2
julia --project=scripts scripts/run_network.jl       # says which part adds its entries; runs nothing
julia --project=scripts scripts/run_symbolic.jl      # says which part adds its entries; runs nothing
julia --project=scripts scripts/figures.jl           # every archive → the PDFs in results/
```

Each takes arguments, so one figure can be regenerated on its own:

```sh
julia --project=scripts scripts/run_spline.jl wave-spline-h0.1
julia --project=scripts scripts/figures.jl wave-spline-h0.1
julia --project=scripts scripts/spline_energy_scan.jl --steps 0.2,0.1
julia --project=scripts scripts/spline_phase_drift_scan.jl
julia --project=scripts scripts/profile_residual.jl --steps 0.05,0.025
```

**The solves and the figures are separate stages, with `runs/` between them.** A figure can be
restyled without paying for the solves again, and a missing figure stays distinguishable from a
failed run.

| file | does |
|:--|:--|
| `archives.jl` | where output goes, the archive schema and its reader and writer, the option parser, the printing helpers. Depends on `JLD2` and `Printf` alone, so a renderer loads it cheaply. No work at top level. |
| `experiments.jl` | the registry: the entries of every family that is merged, as plain data — the problem, its time step and grid, the ansatz resolution and the coupling — plus the ladders of the two scans. It builds nothing and runs nothing on `include`; a family whose part has not merged holds an empty tuple. |
| `spline_model.jl` | the spline family, built: the problem and the method of an entry, one run timed, the archive of a run, and the error of a run against the exact solution at its final time. |
| `run_spline.jl` | the spline family's driver: the smallest entry by default, or the entries named on the command line, each archived under `runs/`. |
| `run_network.jl` | the network family's driver. The registry holds no network entry yet, and the driver says so, naming the part that adds them. |
| `run_symbolic.jl` | the symbolic family's driver, in the same state. |
| `figures.jl` | globs `runs/`, draws each archive as the kind it declares, and writes `results/<stem>.pdf`. **The only place `save` is called.** |
| `spline_energy_scan.jl` | the relative energy error over time at a ladder of time steps, archived as one `"energy"` run that `figures.jl` draws as a comparison. |
| `spline_phase_drift_scan.jl` | the error against the exact solution at the final time, at a ladder of ansatz resolutions. Prints a table and writes nothing. |
| `profile_residual.jl` | one slab's residual, timed, with its allocations and the cost of a full step. |

### No figure code lives here

Figures are built by `MultisymplecticIntegrators.Diagnostics`, in the `MultisymplecticIntegratorsPlots`
extension, which `using CairoMakie` activates. `figures.jl` reads an archive, hands it to the
diagnostics, and saves what comes back:

| want | use |
|:--|:--|
| the field over the whole slab | `Diagnostics.plot_solution` |
| the energy error of several runs over time | `Diagnostics.plot_energy_error` |
| an animation of the field over the stored steps | `Diagnostics.animate_solution` |
| every figure one archive earns, named from its `"kind"` and `"stem"` | `Diagnostics.figures` |

Fonts, line widths and colours come from `Diagnostics.plot_theme()`, which the figure methods
apply themselves, so changing the theme restyles every figure. A figure's own extent is not the
theme's: the extension sizes it from `FIGURE_WIDTH` and `PANEL_HEIGHT` per panel, and the `figsize`
keyword overrides that.

### The archive schema

One JLD2 file per run, `runs/<stem>.jld2`, and two keys carry the schema: `"kind"`, which says what
shape of figure the run draws, and `"stem"`, which names the figure. `store_run!` refuses a data set
without a `"kind"`, and `figures.jl` refuses a kind it cannot draw — a figure guessed at would be
drawn from the wrong keys.

| kind | keys |
|:--|:--|
| `"solution"` | `"label"`, `"problem_label"`, `"timestep"`, `"final_time"`, `"dimension"`, `"t"`, `"x"`, `"u"` (one matrix per component), `"energy_t"`, `"energy_error"` |
| `"energy"` | `"title"`, `"labels"`, `"t"` and `"errors"` (one vector per series) |

### Naming

**One scheme: `<problem>-<family>-h<timestep>`**, so sorting a directory groups the methods of one
problem together and the steps of one method; a run that compares several steps rather than fixing
one is named for its study instead (`wave-spline-energy`). An archive and its figure share the stem,
because the renderer globs the directory rather than consulting a registry.

### Where the registry entries come from

`experiments.jl` holds the spline family. The network and symbolic families are empty on purpose:
their ansätze arrive with their parts, and each of those parts adds its own registry entries rather
than this one guessing at constructors that do not exist yet.
