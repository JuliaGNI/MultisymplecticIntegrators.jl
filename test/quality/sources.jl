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

# The package's module and its submodules, whose docstrings count as the package's.
function package_modules()
    modules = Module[M]
    for name in names(M; all = true)
        isdefined(M, name) || continue
        value = getfield(M, name)
        if value isa Module && value !== M && parentmodule(value) === M
            push!(modules, value)
        end
    end
    return modules
end

# The names of `exported` without a docstring of the package. A docstring is keyed by the binding
# of the function it documents, the binding that a name aliases, and it lives in `Docs.meta` of the
# module that writes it. So a method docstring of the package on another package's function counts,
# and that package's own docstring does not. `names(M)` lists the module's own name, which is not
# an export.
function undocumented_exports(exported = names(M))
    modules = package_modules()
    documented(name) = any(
        m -> haskey(Base.Docs.meta(m), Base.Docs.aliasof(Base.Docs.Binding(M, name))),
        modules)
    return sort([name for name in exported if name !== nameof(M) && !documented(name)])
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

    @testset "S2d every exported name has a docstring of the package" begin
        @test length(names(M)) > 50
        @test isempty(undocumented_exports())
        # SimpleSplines documents `Dirichlet` and the package does not; the package documents a
        # method of GeometricBase's `degree` and its own `xgrid`.
        @test undocumented_exports([:Dirichlet, :degree, :xgrid]) == [:Dirichlet]
    end
end
