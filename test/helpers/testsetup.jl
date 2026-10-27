using MultisymplecticIntegrators
using Test

"""
    test_eltype(x, ::Type{T})

Assert that `x` and every entry of a tuple or array of vectors have element type `T`.
"""
function test_eltype(x::AbstractArray, ::Type{T}) where {T}
    @test eltype(x) === T
    return nothing
end

function test_eltype(x::Tuple, ::Type{T}) where {T}
    for entry in x
        test_eltype(entry, T)
    end
    return nothing
end

test_eltype(x, ::Type{T}) where {T} = @test typeof(x) === T

"""The tolerance of a `Float32` or `Float64` comparison against a closed form."""
tolerance(::Type{T}) where {T} = 16 * eps(T)

"""
    newton_tolerance(::Type{T}, options)

The Newton tolerance of a `§A` reference entry — the `f_abstol` field of the oracle's
`SOLVER_OPTIONS` — carried to the working precision as the same number of `eps(T)`. The value is
read from the `options` the caller passes, so no copy of it stands here.
"""
newton_tolerance(::Type{T}, options) where {T} = T(options.f_abstol / eps(Float64)) * eps(T)

"""
    assert_no_upcast(x, ::Type{T})

Assert that `x` — an array, a named tuple of arrays or a scalar — holds only values of element type
`T`, so that a `Float32` problem stays `Float32` through the solution, the cache and the trace.
"""
function assert_no_upcast(x::AbstractArray, ::Type{T}) where {T}
    @test eltype(x) === T
    return nothing
end

function assert_no_upcast(x::Vector, ::Type{T}) where {T}
    @test eltype(x) === T
    @test all(v -> typeof(v) === T, x)
    return nothing
end

function assert_no_upcast(x::NamedTuple, ::Type{T}) where {T}
    for entry in values(x)
        assert_no_upcast(entry, T)
    end
    return nothing
end

assert_no_upcast(x, ::Type{T}) where {T} = @test typeof(x) === T
