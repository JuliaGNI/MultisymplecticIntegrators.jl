# Release Notes

All notable changes to MultiSymplectic.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually
changed, so that a compat-only bump can be told apart from a rename or a change in results.

This file was started on 2026-08-31 and deliberately holds no entries. Nothing has been
released yet — there are no tags, and `Project.toml` stands at `0.1.0` — so the development
history that predates this file is in `git log` alone. It is named as a gap rather than
reconstructed, because a changelog assembled after the fact loses exactly the reasoning that
makes it worth keeping.

## [Unreleased] — targeting 0.1.0

### New Features

- **Continuous integration.** The repository had no `.github` directory at all; it now carries the
  same `CI.yml`, `CompatHelper.yml`, `Documenter.yml` and `TagBot.yml` as every other repository in
  the tree. **CI is expected to be red** until the blockers under *Open Issues* are dealt with —
  that is the pre-existing state of the package, not a regression.
- **A documentation build.** `docs/` now holds `make.jl`, `Project.toml` and a two-page manual, so
  `Documenter` has something to build and the doctest job has an environment.
- **`test/runtests.jl`, so `Pkg.test()` runs something.** It includes the five files in `test/` that
  actually assert — the three multiplier momentum-carry tests, the bilinear FEM wave integrator and
  the space-time spline wave integrator. The other sixteen files there are numerical experiments
  that integrate for thousands of steps and write `.jld2` archives and figures; they stay
  hand-run. `test_spacetime_spline_wave_int.jl` states its assertions at top level, so the entry
  point supplies the `@testset` it lacks.

### Bug Fixes

### Breaking Changes

- **`tests/` renamed to `test/`.** Neither `Pkg.test()` nor `julia-actions/julia-runtest` looks at
  `tests/`, so the ten scripts there were unreachable by any standard tool. Anything referring to
  the old path by name needs updating.
- **`[compat] julia = "1.10"` added.** There was no Julia bound at all. 1.10 is the LTS and the
  floor across the tree, and the CI matrix resolves its lower entry from this field.
- **`test/Manifest.toml` untracked.** The root one was already removed; this one survived the
  `tests/` → `test/` rename while `.gitignore` claimed to ignore it. The test environment now
  resolves from `test/Project.toml`, so anyone relying on the pinned versions it recorded will get
  a fresh resolve instead.
