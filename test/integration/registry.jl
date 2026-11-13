using Test

# `scripts/experiments.jl` is the registry of the scripts and no part of the package, so it is
# tested through its file.
include(joinpath(@__DIR__, "..", "..", "scripts", "experiments.jl"))

# `scripts/elm_model.jl` is the ELM family's model file and no part of the package either: it
# builds the family and registers it, and the one driver includes it after the registry, so the
# test includes it in the same order.
include(joinpath(@__DIR__, "..", "..", "scripts", "elm_model.jl"))

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

        # The registry keeps its own tuple, so an entry added to the caller's vector later
        # bypasses no check.
        later = [(stem = "d", label = "d")]
        register_family!(:third, later; run = archive)
        push!(later, (stem = "c", label = "another c"))
        @test family_entries(:third) == ((stem = "d", label = "d"),)
    end

    @testset "a family's default entry is the one it names, and else its cheapest" begin
        tied = ((stem = "e-big", label = "e", timestep = 0.1),
            (stem = "e-run", label = "e", timestep = 0.1),
            (stem = "e-fine", label = "e", timestep = 0.05))
        @test register_family!(:defaulted, tied; run = archive, default = "e-big") ===
              :defaulted
        @test default_entry(:defaulted) === tied[1]
        @test smallest_entry(tied) === tied[2]
        @test default_entry(:first) === smallest_entry(entries)
        @test_throws r"^ArgumentError: the default `e-none` of the family `undefaulted` names none" register_family!(
            :undefaulted, ((stem = "f", label = "f"),); run = archive, default = "e-none")
        @test !haskey(FAMILIES, :undefaulted)
    end

    @testset "a row's problem and its label come from one map" begin
        @test matrix_problem_label(:wave) == "Wave"
        @test matrix_problem_label(:vector_wave) == "VectorWave"
        @test matrix_problem_label(:sine_gordon) == "SineGordon"
        @test matrix_problem_label(:linear_transport) == "LinearTransport"
        @test matrix_problem_label(:generated) == "generated"
        problem = matrix_problem("test family", :vector_wave, (:wave, :vector_wave);
            timestep = 0.05)
        @test problem.D == 2
        @test problem.timestep == 0.05
        @test problem.timespan == (0.0, 0.2)
        @test xspan(problem) == (0.0, 1.0)
        @test xstep(problem) == 0.05
        transport = matrix_problem("test family", :linear_transport, (:linear_transport,);
            timestep = 0.1, timespan = (0.0, 0.3), xspan = (-0.5, 0.0), xstep = 0.1)
        @test transport.timespan == (0.0, 0.3)
        @test xspan(transport) == (-0.5, 0.0)
        @test xstep(transport) == 0.1
        @test_throws r"^ArgumentError: the box scheme has no problem `linear_transport`; it takes :wave, :sine_gordon$" matrix_problem(
            "box scheme", :linear_transport, (:wave, :sine_gordon); timestep = 0.1)
    end

    # The family is started from one guess: the previous slab's field fitted by a trained hidden
    # layer, whose target falls back to the trained fit of the initial condition at the first slab.
    @testset "the family is built from its one guess" begin
        method = elm_method(first(family_entries(:elm)))
        guess = MultisymplecticIntegrators.initial_guess(method)
        @test guess isa FitTo
        @test guess.target isa PreviousSlab
        @test guess.method isa TrainedHiddenLayer
    end
end
