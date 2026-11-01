using GeometricBase: datatype, ntime, parameters, timespan, timestep
using GeometricIntegratorsBase
using GeometricSolutions
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using Test

import GeometricIntegratorsBase: cache, current, solver

include("../../helpers/problem_generator.jl")

const M = MultisymplecticIntegrators

"""
    central_differences(int, θ, params)

The central differences of the slab's `residual!` at `θ`, one column per unknown, with the step
`δ_k = cbrt(eps(T)) max(1, |θ_k|)`. Each column divides by the step as it is represented.
"""
function central_differences(int, θ::AbstractVector{T}, params) where {T}
    N = length(θ)
    J = zeros(T, N, N)
    forward = zeros(T, N)
    backward = zeros(T, N)
    x = copy(θ)
    for k in 1:N
        δ = cbrt(eps(T)) * max(one(T), abs(θ[k]))
        x[k] = θ[k] + δ
        M.residual!(forward, x, nothing, params, int)
        up = x[k]
        x[k] = θ[k] - δ
        M.residual!(backward, x, nothing, params, int)
        J[:, k] .= (forward .- backward) ./ (up - x[k])
        x[k] = θ[k]
    end
    return J
end

@testset "A4a the Jacobian is the derivative of the residual, $T" for T in (Float64, Float32)
    entries = [e for e in generate_problems(T; rng = Xoshiro(SEED)) if e.potential ≥ 3]
    rng = Xoshiro(SEED + 2)
    @testset "potential $(entry.potential), $(nameof(typeof(coupling)))" for entry in entries,
        coupling in COUPLINGS

        slab = first_slab(entry, coupling)
        s = solver(slab.int)
        N = length(cache(slab.int).x)
        θ = 2 .* rand(rng, T, N) .- 1
        # the Jacobian Newton calls, with the parameters `residual!` takes
        params = parameters(entry.problem)
        SimpleSolvers.jacobian!(s, θ, (current(slab.solstep), params, slab.int))
        J = copy(SimpleSolvers.jacobianmatrix(s))
        scale = maximum(abs, J)
        @test scale ≥ T(1 // 100)
        @test maximum(abs, J - central_differences(slab.int, θ, params)) ≤
              sqrt(eps(T)) * scale
    end
end
