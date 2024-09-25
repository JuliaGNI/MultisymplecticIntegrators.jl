# raw"""
#     Applying finite difference method to the nonlinear wave equation
#     u_tt(xj,tn) = u_j^{n+1} - 2u_j^n + u_j^{n-1} / Δt^2
#     u_xx(xj,tn) = u_{j+1}^n - 2u_j^n + u_{j-1}^n / Δx^2
#     by defining s = Δt^2 / Δx^2,
#     then we could arrange the equation as
#     u_j^{n+1} = s(u_{j+1}^n + u_{j-1}^n) + 2(1-s)u_j^n - u_j^{n-1} + Δt^2 V'(u_j^n)
#     when s <=1, the scheme is stable
# """

# struct FiniteDifference{DT,NX} <: PDEMethod{DT}
#     Δt::DT
#     Δx::DT
#     function FiniteDifference{DT,NX}(Δt,Δx) where {DT,NX}
#         @assert Δt^2 / Δx^2<=1 "The scheme is unstable"
#         @assert typeof(Δt) == typeof(Δx) "The type of Δt and Δx should be the same"
        
#         new{typeof(Δt),size(û,1)}(Δt,Δx)
#     end
# end

# struct FiniteDifferenceCache{DT,D,NX} <:PDEIntegratorCache{DT,D}
#     u::Vector{DT,D,NX}
#     ū::Vector{DT,D,NX}
#     function FiniteDifferenceCache{DT,D,NX}() where {DT,NX}
#         new(zeros(DT,D,NX),zeros(DT,D,NX),zeros(DT,D,NX))   
#     end
# end

# function GeometricIntegrators.Integrators.reset!(cache::FiniteDifferenceCache, t, û, u)
#     copyto!(cache.ū, u)
#     copyto!(cache.u, û)
# end

# GeometricIntegrators.Integrators.nlsolution(cache::FiniteDifferenceCache) = cache.û

# function GeometricIntegrators.Integrators.Cache{ST}(problem::PDEProblem, method::FiniteDifferenceCache; kwargs...) where {ST}
#     FiniteDifferenceCache{ST, ndims(problem), nnodes(method)}(; kwargs...)
# end

# function GeometricIntegrators.Integrators.integrate_step!(sol,history, cache::FiniteDifferenceCache,int::FiniteDifference,Δt,Δx)
#     nothing
# end



