using MultisymplecticIntegrators
using Test

include("../helpers/testsetup.jl")

@testset "Space-time quadrature" begin
    tbreaks = [0.0, 0.4, 1.0]
    xbreaks = [-0.5, -0.2, -0.05, 0.0]

    @testset "exactness for t^a x^b up to 2n - 1, in Float64 and Float32" begin
        n = 3
        for T in (Float32, Float64)
            q = SpaceTimeQuadrature{T}(n; tbreaks = T.(tbreaks), xbreaks = T.(xbreaks))
            domain = (T(xbreaks[1]), T(xbreaks[end]))
            for a in 0:(2n - 1), b in 0:(2n - 1)

                exact = (1 / (a + 1)) * (domain[2]^(b + 1) - domain[1]^(b + 1)) / (b + 1)
                @test q((t, x)->t^a*x^b)≈exact rtol=tolerance(T) atol=tolerance(T)
                @test typeof(q((t, x) -> t^a * x^b)) === T
            end
        end
    end

    @testset "the element type" begin
        for T in (Float32, Float64)
            q = SpaceTimeQuadrature{T}(2; xbreaks = T[-0.5, -0.2, 0.0],
                tbreaks = T[0.0, 0.5, 1.0])
            test_eltype(q.t̂, T)
            test_eltype(q.ωt, T)
            test_eltype(q.x, T)
            test_eltype(q.ωx, T)
            @test typeof(q((t, x) -> t * x)) === T
            points = boundary_points(q)
            for set in values(points)
                test_eltype(set.t̂, T)
                test_eltype(set.x, T)
            end
            @test q((t, x) -> one(T)) ≈ (1 - 0) * (0 - (-0.5)) rtol = tolerance(T)
        end
    end

    @testset "the four boundary point sets follow xspan" begin
        for xspan in ((-0.5, 0.0), (0.0, 2.0))
            q = SpaceTimeQuadrature{Float64}(2; xbreaks = collect(xspan), tbreaks = tbreaks)
            points = boundary_points(q)
            @test points.bottom.t̂ == [0.0]
            @test points.top.t̂ == [1.0]
            @test points.left.x == [xspan[1]]
            @test points.right.x == [xspan[2]]
            @test points.bottom.x ≈ q.x
            @test points.left.t̂ ≈ q.t̂
        end
    end

    @testset "the rule and its arguments" begin
        q = SpaceTimeQuadrature(2; xbreaks = [0.0, 1.0])
        @test nnodes(q) == (length(q.t̂), length(q.x))
        @test nodes(q) == (q.t̂, q.x)
        @test weights(q) == (q.ωt, q.ωx)
        @test_throws ArgumentError SpaceTimeQuadrature(0; xbreaks = [0.0, 1.0])
        @test_throws ArgumentError SpaceTimeQuadrature(2)
        @test_throws ArgumentError SpaceTimeQuadrature{Float64}(2;
            xbreaks = [0.0, 1.0], tbreaks = [0.0, 0.5])
        @test_throws ArgumentError SpaceTimeQuadrature(2; xbreaks = [1.0, 0.0])
        @test_throws ArgumentError SpaceTimeQuadrature(2; xbreaks = [0.0, 0.0, 1.0])
        @test_throws ArgumentError SpaceTimeQuadrature(2; xbreaks = [0.0])
    end
end
