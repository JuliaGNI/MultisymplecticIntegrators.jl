using Test

# `scripts/experiments.jl` is the registry of the scripts and no part of the package, so it is
# tested through its file.
include(joinpath(@__DIR__, "..", "..", "scripts", "experiments.jl"))

@testset "Registry" begin
    archive(entry) = Dict{String, Any}("kind" => "solution")
    entries = ((stem = "a-h0.2", label = "a", timestep = 0.2),
        (stem = "a-h0.1", label = "a", timestep = 0.1))

    @testset "a family registers once and its entries are found by stem" begin
        @test register_family!(:first, entries; run = archive) === :first
        @test family_entries(:first) === entries
        @test family_run(:first) === archive
        @test entry_by_stem(entries, "a-h0.1") === entries[2]
        @test smallest_entry(entries) === entries[1]
        @test_throws ArgumentError register_family!(:first, entries[1:1]; run = archive)
        @test_throws ArgumentError family_entries(:unknown)
    end

    @testset "a stem names one archive across the registry" begin
        twice = ((stem = "b", label = "b, h = 0.2"), (stem = "b", label = "b, h = 0.1"))
        @test_throws ArgumentError register_family!(:twice, twice; run = archive)
        @test !haskey(FAMILIES, :twice)
        clash = ((stem = "a-h0.2", label = "another family's a"),)
        @test_throws ArgumentError register_family!(:clash, clash; run = archive)
        @test !haskey(FAMILIES, :clash)
        @test register_family!(:second, ((stem = "c", label = "c"),); run = archive) ===
              :second
    end
end
