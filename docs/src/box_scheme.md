```@meta
CurrentModule = MultisymplecticIntegrators
```

# The box scheme

[`BoxScheme`](@ref) is the Preissmann box scheme of a density's first-order form
`K z_t + L z_x = ∇S(z)`, with `z = (u, v, w)`: the baseline that every other method of the package
is compared against. It is a method of its own, not a [`SpaceTimeGalerkin`](@ref), and
[`integrate`](@ref) advances it like any other method. Every row of the form is evaluated at the
centre of a box of the output grid, which is second order in the time step and in the space step
and satisfies the discrete multisymplectic conservation law of the form.

The resolution is the problem's own, one box per `timestep` and `xstep`, and the state lives at the
nodes of the output grid. `Wave`, `VectorWave` and `SineGordon` are in the class of densities the
scheme admits; a density outside it, such as `LinearTransport`'s, is refused when the integrator is
built.

```@example box_scheme
using MultisymplecticIntegrators

problem = Wave.problem(timespan = (0.0, 0.2), timestep = 0.1, xstep = 0.05)
result = integrate(problem, BoxScheme())
result.stats
```

The scheme's energy, which [`energy_error`](@ref) reports, is the density's Hamiltonian at the
midpoints of the boxes, summed with the weights of [`box_quadrature_weights`](@ref).
