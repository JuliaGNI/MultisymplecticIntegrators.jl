cd("MultiSymplectic.jl")
using Pkg
Pkg.activate(".")

using Symbolics

const N = 256
l = 2π
Δx = l/(N-1)


@variables u v
uₕ = Symbolics.variables(:u, 1:N)
vₕ = Symbolics.variables(:v, 1:N)
Hₙ = sum(1/2 * vₕ[i]^2 + 1/2 *(uₕ[i] - uₕ[i-1])^2/ Δx ^2 for i in 2:N) + 1/2 * vₕ[1]^2 + 1/2 *(uₕ[1] - uₕ[N])^2/ Δx ^2

function fᵢ(i,uh,vh,dx)
    ((uh[i+1]-2uh[i]+uh[i-1])/dx^2, vh[i],)
end

typeof(fᵢ(2,uₕ,vₕ,Δx))

vector_f = [fᵢ(i,uₕ,vₕ,Δx) for i in 2:N-1]
pushfirst!(vector_f, ((uₕ[N]-2uₕ[1]+uₕ[2])/Δx^2, vₕ[1],))
push!(vector_f, ((uₕ[1]-2uₕ[N]+uₕ[N-1])/Δx^2, vₕ[N],))