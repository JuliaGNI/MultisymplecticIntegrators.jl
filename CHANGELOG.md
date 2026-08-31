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

### Bug Fixes

### Breaking Changes

- **`tests/` renamed to `test/`.** Neither `Pkg.test()` nor `julia-actions/julia-runtest` looks at
  `tests/`, so the ten scripts there were unreachable by any standard tool. Anything referring to
  the old path by name needs updating.
- **`[compat] julia = "1.10"` added.** There was no Julia bound at all. 1.10 is the LTS and the
  floor across the tree, and the CI matrix resolves its lower entry from this field.

## Open Issues

Four blockers keep this package from loading or testing. Full detail, and the order to work in, in
`~/Research/Tasks/Revive MultiSymplectic.md`. Recorded 2026-08-31:

- No `test/runtests.jl` entry point — `test/` is ten ad-hoc scripts.
- No `[extras]` and no `[targets]`, so there is no test environment to resolve.
- Stale bounds: `GeometricIntegratorsBase = "0.1.11"` against 0.6.4, `SimpleSolvers = "0.7.8"`
  against 0.13.2. Only 10 of 31 dependencies are bounded at all.
- `Gtk4`, `ProfileView`, `PProf`, `Infiltrator`, `Revise` and `Plots` are hard dependencies; a
  headless runner should not install a GUI toolkit to run tests.

Separately, `src/utils/common.jl` is one of two files in the tree that **JuliaFormatter cannot
process** — four flat 64-element `Float64` literals, longest line 1624 characters, which take the
Julia process down rather than raising a catchable error. It will block the pre-commit hook if ever
staged.
