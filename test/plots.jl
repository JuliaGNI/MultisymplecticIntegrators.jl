using CairoMakie
using MultisymplecticIntegrators
using Test

using GeometricBase: ntime

# The stubs live in the package's `Diagnostics` submodule, so the methods the extension adds are
# reached the way a caller reaches them: `plot_solution` and `plot_energy_error` are names
# `GeometricProblems` exports too.
import MultisymplecticIntegrators.Diagnostics as Diagnostics

include("helpers/testsetup.jl")

# Five steps of the wave equation. Loading CairoMakie above is what activates the `Makie` weak
# dependency, and with it the extension: the stub alone has no method.
function five_step_wave()
    problem = Wave.problem(timespan = (0.0, 0.25), timestep = 0.05, xstep = 0.05)
    ansatz = TensorBSplineAnsatz(degree = 2, ncells = (2, 20), xspan = (0.0, 1.0), n = 3)
    method = SpaceTimeGalerkin(ansatz, CoefficientRestriction())
    return problem, method, integrate(problem, method)
end

@testset "Plotting" begin
    problem, method, result = five_step_wave()
    err = energy_error(problem, method, result.traces)
    times = [result.solution.t[n] for n in 1:length(err)]
    directory = mktempdir()

    try
        @testset "P1 the three figures are drawn" begin
            solution_path = joinpath(directory, "solution.pdf")
            figure = Diagnostics.plot_solution(problem, result;
                path = solution_path, title = "Wave")
            @test figure isa Figure
            @test isfile(solution_path)
            @test filesize(solution_path) > 0

            energy_path = joinpath(directory, "energy.pdf")
            energy = Diagnostics.plot_energy_error([times, times], [err, 2 .* err];
                labels = ["h = 0.05", "2 h"], path = energy_path, title = "Energy error")
            @test energy isa Figure
            @test isfile(energy_path)
            @test filesize(energy_path) > 0

            animation_path = joinpath(directory, "solution.mp4")
            animation = Diagnostics.animate_solution(problem, result; path = animation_path)
            @test animation isa Figure
            @test isfile(animation_path)
            @test filesize(animation_path) > 0
        end

        @testset "P1 one theme, read by every figure" begin
            @test Diagnostics.plot_theme() isa Theme
            @test Diagnostics.plot_theme().fontsize[] == 18
            @test Diagnostics.plot_theme().Axis.xlabelsize[] == 22
            @test Diagnostics.plot_theme().Lines.linewidth[] == 2

            # The figure carries the theme's sizes, not Makie's defaults: a method that drew
            # without applying it would show the defaults on its axes and its lines.
            figure = Diagnostics.plot_solution(problem, result)
            axes = filter(x -> x isa Axis, figure.content)
            @test !isempty(axes)
            @test all(ax -> ax.xlabelsize[] == 22, axes)
            @test all(ax -> ax.ylabelsize[] == 22, axes)
            @test all(ax -> ax.xticklabelsize[] == 16, axes)
            @test all(ax -> ax.titlesize[] == 20, axes)

            # Two energy series take their colours from the theme's cycle, in its order, and their
            # width from the theme as well.
            energy = Diagnostics.plot_energy_error(times, [err, err]; labels = ["a", "b"])
            energy_axes = filter(x -> x isa Axis, energy.content)
            @test length(energy_axes) == 1
            palette = Diagnostics.plot_theme().palette.color[]
            colours = [plot.color[] for plot in energy_axes[1].scene.plots]
            @test colours == palette[1:2]
            @test all(plot -> plot.linewidth[] == 2, energy_axes[1].scene.plots)
        end

        @testset "P1 a figure from an archive" begin
            # The renderer reads plain vectors out of an archive, not a solution object, so the
            # drawing is reachable without a run. This is the shape `scripts/archives.jl` writes.
            nx = length(MultisymplecticIntegrators.xgrid(problem))
            steps = ntime(result.solution)
            archive = Dict{String, Any}(
                "kind" => "solution",
                "stem" => "wave-spline-h0.05",
                "label" => "spline, h = 0.05",
                "problem_label" => "Wave",
                "t" => [result.solution.t[n] for n in 0:steps],
                "x" => collect(MultisymplecticIntegrators.xgrid(problem)),
                "u" => [[result.solution.u[n][j]
                         for n in 0:steps, j in 1:nx]],
                "dimension" => 1)
            path = joinpath(directory, "archive-solution.pdf")
            figure = Diagnostics.plot_solution(archive; path = path)
            @test figure isa Figure
            @test isfile(path) && filesize(path) > 0

            energy_archive = Dict{String, Any}(
                "kind" => "energy",
                "stem" => "wave-spline-energy",
                "title" => "Energy error against time",
                "t" => times,
                "labels" => ["h = 0.05"],
                "errors" => [err])
            energy_path = joinpath(directory, "archive-energy.pdf")
            drawn = Diagnostics.plot_energy_error(energy_archive; path = energy_path)
            @test drawn isa Figure
            @test isfile(energy_path) && filesize(energy_path) > 0
        end

        @testset "P1 the archive's kind picks the figure" begin
            # The renderer's one decision: which figure an archive earns. A kind this module cannot
            # draw is refused rather than drawn from the keys of another kind, because a figure
            # made from the wrong vectors looks like a result.
            nx = length(MultisymplecticIntegrators.xgrid(problem))
            steps = ntime(result.solution)
            archive = Dict{String, Any}(
                "kind" => "solution", "stem" => "wave-spline-h0.05",
                "t" => [result.solution.t[n] for n in 0:steps],
                "x" => collect(MultisymplecticIntegrators.xgrid(problem)),
                "u" => [[result.solution.u[n][j] for n in 0:steps, j in 1:nx]])
            energy_archive = Dict{String, Any}(
                "kind" => "energy", "stem" => "wave-spline-energy",
                "t" => times, "labels" => ["h = 0.05"], "errors" => [err])

            solution_drawn = only(Diagnostics.figures(archive))
            @test first(solution_drawn) == "wave-spline-h0.05"
            @test last(solution_drawn) isa Figure
            # a space-time figure: no logarithmic axis anywhere in it
            @test !any(ax -> ax isa Axis && ax.yscale[] === log10, last(solution_drawn).content)

            energy_drawn = only(Diagnostics.figures(energy_archive))
            @test first(energy_drawn) == "wave-spline-energy"
            @test last(energy_drawn) isa Figure
            # the energy figure's one axis is logarithmic, which is what distinguishes it
            @test any(ax -> ax isa Axis && ax.yscale[] === log10, last(energy_drawn).content)

            @test_throws ArgumentError Diagnostics.figures(
                Dict{String, Any}("kind" => "phase-portrait", "stem" => "x"))
            @test_throws ArgumentError Diagnostics.figures(Dict{String, Any}("stem" => "x"))
            @test_throws ArgumentError Diagnostics.figures(Dict{String, Any}("kind" => "solution"))
        end

        @testset "P1 a series with nothing to draw" begin
            # The first energy error is zero and a logarithmic axis cannot take it, so the point is
            # dropped from the time vector too; a series of nothing but zeros is left out rather
            # than drawn empty.
            figure = Diagnostics.plot_energy_error(times, [err, zeros(length(err))];
                labels = ["a", "zero"])
            axes = filter(x -> x isa Axis, figure.content)
            @test length(axes[1].scene.plots) == 1
            @test_throws ArgumentError Diagnostics.plot_energy_error(times, [err];
                labels = ["one", "two"])
        end
    finally
        rm(directory; recursive = true, force = true)
    end
end
