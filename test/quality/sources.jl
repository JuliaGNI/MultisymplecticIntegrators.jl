using MultisymplecticIntegrators
using Test

const M = MultisymplecticIntegrators

# The text of every source file of the package, by its path under `src/`.
function source_texts()
    src = joinpath(pkgdir(M), "src")
    texts = Dict{String, String}()
    for (root, _, names) in walkdir(src), name in names

        endswith(name, ".jl") || continue
        file = joinpath(root, name)
        texts[relpath(file, src)] = read(file, String)
    end
    return texts
end

# The text of every docstring the package's module defines.
function docstring_texts()
    texts = String[]
    for (_, multidoc) in Base.Docs.meta(M), (_, doc) in multidoc.docs

        push!(texts, join(string.(doc.text)))
    end
    return texts
end

# The files whose text holds `pattern`; the failure of an `isempty` test lists them.
function offenders(pattern)
    sort([file for (file, text) in source_texts() if occursin(pattern, text)])
end

@testset "Sources and docstrings" begin
    @testset "S2a no forwarding of properties" begin
        @test isempty(offenders("getproperty"))
    end

    @testset "S2b no docstring cites the plan" begin
        texts = docstring_texts()
        @test length(texts) > 50
        @test isempty(filter(text -> occursin('§', text) || occursin("the plan", text), texts))
    end

    @testset "S2c one docstring of is_linear_in_coefficients" begin
        binding = Base.Docs.Binding(M, :is_linear_in_coefficients)
        @test length(Base.Docs.meta(M)[binding].docs) == 1
    end

    @testset "S2d the space-time quadrature is gone" begin
        for name in (:SpaceTimeQuadrature, :boundary_points)
            @test !isdefined(M, name)
            @test isempty(offenders(string(name)))
        end
    end
end
