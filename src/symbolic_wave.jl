"""
    symbolic_wave.jl
    Implementation of linear wave equation with symbolic computation. Grid Based method. Compose a system of discretized Hamilronian ODE.
"""

cd("MultiSymplectic.jl")
using Pkg
Pkg.activate(".")

using Symbolics

N = 11
l = 2
Δx = l/(N-1)


u = Symbolics.variables(:u, 1:N)
v = Symbolics.variables(:v, 1:N)

z = u .+ v
z = substitute(z, Dict([(u[i] => i) for i in 1:N],[(v[i] => 2i) for i in 1:N]))
z = substitute(z, Dict([(v[i] => 2i) for i in 1:N]))
simplify(z)


# Discrete Hamiltonian, Periodic BC 
Hₙ = sum(1/2 * vₕ[i]^2 + 1/2 *(uₕ[i] - uₕ[i-1])^2/ Δx ^2 for i in 2:N) + 1/2 * vₕ[1]^2 + 1/2 *(uₕ[1] - uₕ[N])^2/ Δx ^2

#Righthand side of the Hamiltonian ODE
function ∂H∂uᵢ(i,uh,vh,dx)
    (uh[i+1]-2uh[i]+uh[i-1])/dx^2
end

function ∂H∂vᵢ(i,uh,vh,dx)
    vh[i]
end


#Deal with the periodic boundary conditions
∂H∂u = [∂H∂uᵢ(i,u,v,Δx) for i in 2:N-1]
pushfirst!(∂H∂u, (u[N]-2u[1]+u[2])/Δx^2)
push!(∂H∂u, (u[1]-2u[N]+u[N-1])/Δx^2)

∂H∂v = [∂H∂vᵢ(i,u,v,Δx) for i in 2:N-1]
pushfirst!(∂H∂v, v[1])
push!(∂H∂v, v[N])


#Initial condition
function initial_position(N,l)
    x = range(0,l,length=N)
    2 * exp.(-((x .- l/2).^2))
end

function initial_velocity(N,l)
    x = range(0,l,length=N)
    2*exp.(-((x .- l/2).^2)) .* 2 .* (-(x .- l/2))
end

u0 = initial_position(N,l)
v0 = initial_velocity(N,l)

# Explicit Euler
# substitute the initial condition into the vector field
# u0x = [substitute(u0, Dict(x => (i-1)*Δx, l =>2)) for i in 1:N]
# J∂H∂z = [substitute(J∂H∂z[j],Dict([uₕ[i] => u0x[i] for i in 1:N])) for j in 1:2N]

# v0x = [substitute(v0, Dict(x => (i-1)*Δx, l =>2)) for i in 1:N]
# J∂H∂z = [substitute(J∂H∂z[j],Dict([vₕ[i] => v0x[i] for i in 1:N])) for j in 1:2N]

    # update 

u = Symbolics.variables(:q, 1:N)
v = Symbolics.variables(:v, 1:N)

u .+= s .* ∂H∂v
v .+= s .* (- ∂H∂u)

# Substitute u with initial conditions
substitute(u[1], Dict([(v[i] => v0[i]) for i in 1:N]))
[substitute(u[j], Dict([(u[i] => u0[i]) for i in 1:N])) for j in 1:N]




v_float = [substitute(v[j], Dict([(v[i] => v0[i]) for i in 1:N])) for j in 1:N]
v_float = [substitute(v1_float[j], Dict([(u[i] => u0[i]) for i in 1:N])) for j in 1:N]

