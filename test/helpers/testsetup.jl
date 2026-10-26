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
