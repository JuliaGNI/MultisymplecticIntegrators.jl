# MultisymplecticIntegrators.jl

implementation of multi-symplectic schemes in julia


## Development

> **This package does not currently load or test.** Its `[compat]` bounds predate several of its
> dependencies, it has no `test/runtests.jl` entry point and no `[targets]` test section, and it
> carries GUI and development packages as hard dependencies. CI is therefore expected to be red
> until that is dealt with; the hooks below will block a commit that stages a `.jl` file for the
> same reason.

### Git hooks

Two hooks live in `.githooks`. They are **not active in a fresh clone** — `core.hooksPath` is local
configuration and does not travel with a push — so enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** acts on **staged `.jl` files only**, and exits immediately when a commit stages
none, so a documentation- or workflow-only commit is not slowed down by it:

- **JuliaFormatter `--check`**, honouring this repository's own `.JuliaFormatter.toml` — **blocks**
  the commit. Formatting is mechanical and always fixable.
- **`fatou lint`**, when `fatou` is installed — **advisory only**, and deliberately so: its
  `unused-import` rule does not follow `include`, so it flags the load-bearing imports of every
  module file.
- **`using <Package>`**, which catches a syntax error or a broken `include` — **blocks**.

**`pre-push`** runs the full test suite with `--check-bounds=auto`, but **only when pushing to
`main` or `master`**; a topic branch is left to CI. It prints nothing for **10–30 minutes**, which
looks exactly like a network hang and is not one. If you do interrupt it, check for an orphaned
Julia process that the killed hook left behind.

Either hook can be bypassed for a single command with `--no-verify`, for a change you know it does
not apply to:

```sh
git commit --no-verify
git push --no-verify
```

Note that `src/utils/common.jl` is a file **JuliaFormatter cannot process** — it takes the Julia
process down rather than raising an error — so staging it will make the `pre-commit` hook fail with
a misleading "not formatted" message.

The hooks are generated from one shared copy and are byte-identical across the related
repositories, so edit them there rather than here — a local edit is silently undone by the next
install.
