using JLD2
using Test

# `scripts/archives.jl` is the writer and reader of the scripts' archives and no part of the
# package, so it is tested through its file, against a temporary runs directory.
include(joinpath(@__DIR__, "..", "..", "scripts", "archives.jl"))

@testset "Archives" begin
    directory = mktempdir()

    try
        @testset "atomic_write writes through a dotted sibling and leaves nothing beside" begin
            path = joinpath(directory, "atomic", "figure.pdf")
            temporaries = String[]
            returned = atomic_write(path) do temporary
                push!(temporaries, temporary)
                write(temporary, "first")
            end
            @test returned == path
            @test read(path, String) == "first"
            temporary = only(temporaries)
            @test dirname(temporary) == dirname(abspath(path))
            @test startswith(basename(temporary), ".")
            @test endswith(temporary, ".pdf")
            @test readdir(dirname(path)) == ["figure.pdf"]

            # A writer that throws leaves the earlier file whole and its temporary removed, and its
            # exception reaches the caller.
            @test_throws ErrorException atomic_write(path) do temporary
                write(temporary, "half")
                error("interrupted")
            end
            @test read(path, String) == "first"
            @test readdir(dirname(path)) == ["figure.pdf"]
        end

        RUNS_DIR[] = joinpath(directory, "runs")

        @testset "store_run! refuses a stem that load_runs would not find" begin
            for stem in ("", "nested/run", ".hidden")
                @test_throws ArgumentError store_run!(stem, Dict{String, Any}("kind" => "energy"))
            end
            @test_throws ArgumentError store_run!("no-kind", Dict{String, Any}())
            # Each is refused before anything is written.
            @test !ispath(RUNS_DIR[])
        end

        @testset "load_runs reads what store_run! wrote and skips a dotted name" begin
            data = Dict{String, Any}("kind" => "energy", "t" => [0.0, 1.0])
            path = store_run!("whole", data)
            @test path == joinpath(RUNS_DIR[], "whole.jld2")
            # The two files a killed writer can leave: a temporary that holds a whole archive and one
            # that holds a truncated one. Neither is a run, and neither is reported unreadable.
            cp(path, joinpath(RUNS_DIR[], ".jl_whole.jld2"))
            write(joinpath(RUNS_DIR[], ".jl_truncated.jld2"), "not an archive")
            runs, unreadable = load_runs()
            @test [run["stem"] for run in runs] == ["whole"]
            @test only(runs)["kind"] == "energy"
            @test only(runs)["t"] == [0.0, 1.0]
            @test isempty(unreadable)
        end
    finally
        rm(directory; recursive = true, force = true)
    end
end
