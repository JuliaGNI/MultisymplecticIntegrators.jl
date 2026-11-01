```@meta
CurrentModule = MultisymplecticIntegrators
```

# The slab seam

A [`SpaceTimeGalerkin`](@ref) method holds three parts: an ansatz, a coupling and an initial
guess. The core of the integrator — [`residual!`](@ref), [`update!`](@ref) and
[`integrate_step!`](@ref) — knows none of them by type. It reaches each part through the
functions on this page, and each part keeps its own data in a block of the slab cache.

## The cache

The [`SpaceTimeGalerkinCache`](@ref) holds what every method fills, whatever its parts:

| field | what it holds | filled by |
|:--|:--|:--|
| `x` | the solver's vector of the slab | the guess, then the Newton solver |
| `u`, `v`, `w` | the field and its derivatives at the points of the slab's quadrature | the ansatz, then the coupling's transform |
| `du`, `dv`, `dw` | the density's derivatives at those points | the core |
| `trace_u`, `trace_v`, `trace_w` | the field at the top of the last slab, at the nodes of the space quadrature | the ansatz, then the coupling's transform |
| `iterations`, `residuals`, `initial_residuals`, `converged` | the record of every step's Newton solve | the core |
| `steps` | the number of steps taken in the run | the core |
| `ansatz` | the ansatz's block | the ansatz |
| `coupling` | the coupling's block | the coupling |

The guess keeps nothing in the cache: it allocates its scratch once per step.

## The ansatz

The ansatz discretises the field on one slab, and owns the quadrature of the slab. Its block is
built by [`ansatz_data`](@ref), whose type [`ansatz_data_type`](@ref) gives without building it.
For a [`TensorBSplineAnsatz`](@ref) the block is a [`TensorBSplineSlab`](@ref): the coefficient
array, its weak form, the top row of the last slab, the tables of the slab's quadrature, and the
space basis at the output grid and at the trace nodes.

The core calls these functions of the ansatz, each on its block:

- [`quadrature_points`](@ref): the time and the space nodes of the slab's quadrature;
- [`slab_field!`](@ref): the field and its derivatives at those nodes, from the solver's vector and
  the coupling's fixed bottom row;
- [`slab_weak_form!`](@ref) and [`free_residual!`](@ref): the weak form of the density's
  derivatives, and its rows of the free coefficients in the solver's layout;
- [`keep_top!`](@ref): keep the top row of the solved slab;
- [`top_values!`](@ref), with [`output_evaluation`](@ref), [`trace_evaluation`](@ref) and
  [`evaluation_points`](@ref): the field at the top of the slab, at the output grid and at the
  trace nodes, from one evaluation;
- [`check_ansatz`](@ref): the refusal of an ansatz that the problem cannot support.

The counts [`ntime_functions`](@ref) and [`nspace_functions`](@ref) size the solver's vector.

## The coupling

The coupling joins consecutive slabs: it fixes the datum of a slab, lays out the free rows of the
solver's vector, adds its terms and its own equations to the residual, and may transform the field.
Its block is a [`CouplingData`](@ref) that carries the coupling itself, built by
[`coupling_data`](@ref), with its type from [`coupling_data_type`](@ref). A coupling that keeps
nothing gets a [`NoCouplingData`](@ref); [`CoefficientRestriction`](@ref) keeps a
[`RestrictionData`](@ref), its bottom row and its momentum load; the weak couplings keep a
[`MultiplierData`](@ref), their multipliers and the tables of their trace basis.

Before a block exists, the core asks the coupling itself:

- [`free_rows`](@ref): the free time rows, declared by every coupling, with no default;
- [`coupling_unknowns`](@ref): the number of the coupling's own unknowns;
- [`check_coupling`](@ref): the refusal of an ansatz that the coupling cannot support;
- [`coupling_internal`](@ref): the coupling's entries of the solution step's internal state.

Within a step, every function of the coupling dispatches on its block:

- [`slab_bottom!`](@ref): fix the datum of the slab;
- [`bottom_datum`](@ref): the bottom row the coupling fixes, or `nothing`;
- [`transform_field!`](@ref): the coupling's transform of the field, wherever the core evaluates
  the field — in the residual, the trace and the solution; the default leaves the field as it is;
- [`coupling_term!`](@ref): the coupling's terms of the residual, with no default, so that a
  coupling without a term declares its no-op;
- [`coupling_guess!`](@ref): the guess of the coupling's own unknowns;
- [`coupling_update!`](@ref): what the coupling reads out of the solved slab and hands on;
- [`copy_coupling_data!`](@ref): the datum, from the cache of the problem's element type into the
  cache the solver works in.

## The guess

The initial guess fills the solver's vector before each solve, through [`initial_guess!`](@ref):
[`Continuation`](@ref) repeats the previous slab's top row, and [`FitTo`](@ref) evaluates a
[`SlabTarget`](@ref) on the slab with [`evaluate!`](@ref) and fits the ansatz to it with a
[`SlabFitMethod`](@ref) through `fit!`, whose contract [`Projection`](@ref) states. The coupling
then guesses its own unknowns.

## A step

[`integrate_step!`](@ref) runs one slab: [`slab_bottom!`](@ref), the guess, the Newton solve, the
record of the solve, and [`update!`](@ref), which evaluates the solved slab, writes the solution and
the trace, and calls [`coupling_update!`](@ref). [`integrate`](@ref) drives the steps and returns
the solution, the traces and the [`solve_stats`](@ref) of the run.

## The action form (U17)

The interior weak form of a slab is the gradient of its discrete action with respect to the
parameters `ĉ` of the ansatz, the whole coefficient array for the tensor B-spline ansatz, and the
residual keeps the rows of it that the free rows carry. The action is the sum over the nodes of the
slab's quadrature of the weighted density `L(u, v, w)`, where `u`, `v` and `w` are the field and its
derivatives that `ĉ` gives at the nodes. By the chain rule, the gradient is the pullback of the
density's derivatives `∂u`, `∂v`, `∂w` at the nodes through the map from `ĉ` to the field there.
The core already evaluates the residual in that order, one seam function per factor:

1. [`slab_field!`](@ref) applies the field map: it builds `ĉ` from the solver's vector and the
   coupling's bottom row, and gives the field at the nodes;
2. [`transform_field!`](@ref) applies the coupling's transform; a lift `ℓ(t̂, x)` that does not
   depend on `ĉ` changes the point at which the density is evaluated and not the pullback;
3. the core evaluates `∂u`, `∂v` and `∂w` of the density at the nodes;
4. [`slab_weak_form!`](@ref) pulls them back through the field map, with the quadrature weights;
5. [`free_residual!`](@ref) writes the rows of the free coefficients in the solver's layout.

For the tensor B-spline ansatz the field map is linear in `ĉ`, so its pullback is the product with
the tabulated basis, which is the weak form in `src/core/weak_form.jl`. An ansatz whose field map
is not linear, such as a network, implements [`slab_field!`](@ref) as its map and
[`slab_weak_form!`](@ref) as the pullback of `∂u`, `∂v` and `∂w` through that map by automatic
differentiation, with [`free_residual!`](@ref) on its own block. A coupling may compute
[`coupling_term!`](@ref) as the gradient, by automatic differentiation, of its own action terms:
the function receives the solver's vector `x` and the coupling's block, which is all those terms
read. Neither the core nor the existing couplings change for either.

The variant that the seam carries is the pullback of `∂L` at the nodes: the core evaluates the
density's derivatives, and the ansatz differentiates only its field map. The seam does not
differentiate `L` inside the ansatz's block, so an ansatz never sees the density, and the density
keeps the derivatives that its problem defines.
