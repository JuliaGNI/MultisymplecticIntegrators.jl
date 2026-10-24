using Test
using MultisymplecticIntegrators

# Ten nodes and ten weights, at the indices `round.(Int, range(1, k; length = 10))`, of the
# 64- and 128-node Gauss–Legendre tables on [0, 1] that `src/utils/common.jl` held at 294f023.
const TABLES = Dict(
    64 => (
        nodes = [0.0003474791321139148, 0.035215413934030215, 0.12359004636973403,
            0.25529842714647355, 0.41517778978800357, 0.5848222102119964,
            0.7447015728535265, 0.876409953630266, 0.9647845860659698, 0.9996525208678861],
        weights = [0.0008916403608482165, 0.008975857887848672, 0.016028964177425775,
            0.021236757561826795, 0.023999694298229155, 0.023999694298229155,
            0.021236757561826795, 0.016028964177425775, 0.008975857887848672,
            0.0008916403608482165]),
    128 => (
        nodes = [8.755602643406579e-5, 0.03216280586104181, 0.11851283497795262,
            0.24912022043192777, 0.4088328470073314, 0.5911671529926685,
            0.7508797795680722, 0.8814871650220474, 0.9678371941389582, 0.9999124439735659],
        weights = [0.00022469048014604518, 0.004313188899308375, 0.007901864329699674,
            0.010573823234110675, 0.012019084340512026, 0.012019084340512026,
            0.010573823234110675, 0.007901864329699674, 0.004313188899308375,
            0.00022469048014604518])
)

@testset "composite_quadrature(1, $k) matches the Gauss–Legendre table" for k in (64, 128)
    rule = MultisymplecticIntegrators.composite_quadrature(1, k)
    idx = round.(Int, range(1, k; length = 10))
    @test length(rule.nodes) == k
    @test length(rule.weights) == k
    @test all(abs.(rule.nodes[idx] .- TABLES[k].nodes) .<= 4eps())
    @test all(abs.(rule.weights[idx] .- TABLES[k].weights) .<= 4eps())
end
