
# using NonlinearSolve

# include("/Users/zeyuanli/Documents/Codes/GeometricProblems.jl/src/sine_gordon.jl")
# include("/Users/zeyuanli/Documents/Codes/MultiSymplectic.jl/src/common.jl")
# include("/Users/zeyuanli/Documents/Codes/MultiSymplectic.jl/src/linear_wave.jl")


# l = NonlinearWave.l
# nx = 256
# Δx = l / nx
# domain = [(0.0, l)]  # 1D domain from (0,0) to (1,1)
# x_grid = range(0, l, length=nx)

# Δt = 0.01
# nt = 50
# tspan = (0.0, Δt * nt)
# s = Δt^2 / Δx^2
# ComDomain = PhysicalDomain(domain, Δx, tspan, Δt)

# nx = LinearWave.Ñ + 2
# Δx = 1. / (LinearWave.Ñ + 1)
# nt = 5
# Δt = LinearWave.tstep
# s = Δt^2 / Δx^2

# x_grid = range(0, 1, length=LinearWave.Ñ + 2)


# sol = zeros(nt, nx)


# sol[1, :] = LinearWave.q₀
# sol[2, :] = LinearWave.q₀ + Δt * LinearWave.p₀


# sol[1, :] = NonlinearWave.initial_position.(x_grid, 0, 0)
# sol[2, :] = NonlinearWave.initial_position.(x_grid, 0, 0) + Δt * NonlinearWave.initial_velocity.(x_grid, 0, 0)


# # sol[1, :] = sin.(pi.* x_grid)
# # sol[2, :] = sin.(pi.* x_grid)

# sol[:, 1] .= 0
# sol[:, end] .= 0

# # finite_difference_integrate!(sol,i, Δt,Δx, NonlinearWave.df)



# # function finite_difference_integrate!(cache::SolutionCache, Δt,nt,Δx,vector_field)
# #     local s = Δt^2/Δx^2

# #     for n in 3:nt
# #         cache.history[n][2:end-1] = s*(cache.history[n-1][3:end] + cache.history[n-1][1:end-2]) + 2*(1-s)*cache.history[n-1][2:end-1] - cache.history[n-2][2:end-1] + 0* Δt^2*vector_field.(cache.history[n-1][2:end-1])
# #     end
# # end

# function f(du, û, p)
#     nx = length(û)

#     du[1] = (1 / (2 * Δt^2)) * (-û[1] + 2p[2, 2] - p[1, 2]) + (1 / (4 * Δt^2)) * (-û[2] + 2p[2, 3] - p[1, 3]) + (1 / (4 * Δx^2)) * (û[2] - 2 * û[1]) + (1 / (2 * Δx^2)) * (p[2, 3] - 2 * p[2, 2]) + (1 / (4 * Δx^2)) * (p[1, 3] - 2 * p[1, 2])
#     for i in 2:nx-1
#         du[i] = (1 / (4 * Δt^2)) * (-û[i-1] + 2p[2, i] - p[1, i]) + (1 / (2 * Δt^2)) * (-û[i] + 2p[2, i+1] - p[1, i+1]) + (1 / (4 * Δt^2)) * (-û[i+1] + 2p[2, i+2] - p[1, i+2])
#         du[i] += (1 / (4 * Δx^2)) * (û[i+1] - 2 * û[i] + û[i-1]) + (1 / (2 * Δx^2)) * (p[2, i+2] - 2 * p[2, i+1] + p[2, i]) + (1 / (4 * Δx^2)) * (p[1, i+2] - 2 * p[1, i+1] + p[1, i])
#         # b[i]+= 1/4 * NonlinearWave.df(1/4 * (p[2,i+1]+p[2,i] + û[i+1]+û[i]))
#         # b[i]+= 1/4 * NonlinearWave.df(1/4 * (p[2,i-1]+p[2,i] + û[i-1]+û[i]))
#         # b[i]+= 1/4 * NonlinearWave.df(1/4 * (p[2,i-1]+p[2,i] + p[1,i-1]+p[1,i]))
#         # b[i]+= 1/4 * NonlinearWave.df(1/4 * (p[2,i+1]+p[2,i] + p[1,i+1]+p[1,i]))
#     end
#     du[end] = (1 / (4 * Δt^2)) * (-û[end-1] + 2p[2, end-2] - p[1, end-2]) + (1 / (2 * Δt^2)) * (-û[end] + 2p[2, end-1] - p[1, end-1]) + (1 / (4 * Δx^2)) * (-2 * û[end] + û[end-1]) + (1 / (2 * Δx^2)) * (-2 * p[2, end-1] + p[2, end-2]) + (1 / (4 * Δx^2)) * (-2 * p[1, end-1] + p[1, end-2])
#     nothing
# end

# s = Δt^2 / Δx^2
# @assert s <= 1 "The scheme is unstable"

# @time for n = 3:nt
#     u0 = sol[n-1, 2:end-1]
#     p = sol[n-2:n-1, :]
#     prob = NonlinearProblem(f, u0, p)
#     û = solve(prob,reltol = 1e-12, abstol = 1e-12)
#     sol[n, 2:end-1] = û
#     sol[n, 1] = 0
#     sol[n, end] = 0
# end


# # finite_difference_integrate!(sol,i, Δt,Δx, NonlinearWave.df)

# multisymplectic_integrate!(sol, Δt, nt, Δx, system_matrix)


# # f(u,p,x2) = [u[1] + u[2]-p[1], u[1]^2 + u[2]^2 - 3*p[2]]
# # u0 = [1.0, 3.0]
# # p = [1.0, 1.0]
# # prob = NonlinearProblem(f,u0,p)
# # sol = solve(prob)


# sol[3:n]


# using Plots
# @gif for i in 1:nt
#     plot(0:1/(nx-1):1, sol[i, :], ylims=(-5, 5))
# end

# plot(0:1/(nx-1):1, sol[1, :])


# sol_v = zeros(nt, nx)
# sol_v[1, :] = NonlinearWave.initial_velocity.(x_grid, 0, 0)
# sol_v[2:nt,:] = (sol[2:nt,:] - sol[1:nt-1,:])/Δt

# sol_w = zeros(nt, nx)
# sol_w[:, 1] .= 0
# sol_w[:, 2:end] = (sol[:, 2:end] - sol[:, 1:end-1])/Δx

# ham_record = zeros(nt)
# for i in 1:nt
#     ham_record[i] = sum(sol_v[i, :] .^ 2) / 2 + sum(sol_w[i, 2:end] .^ 2) / 2
#     ham_record[i] *= Δx
# end

# plot(ham_record[3:end])

using Pkg
cd("IntegratorNN")
Pkg.activate(".")

using GeometricIntegrators
using GeometricProblems:HarmonicOscillator

initial_hamiltonian = HarmonicOscillator.hamiltonian(0.0,HO.ics.q,HO.ics.p,HO.parameters)

HO = HarmonicOscillator.podeproblem(tspan = (0,1000),tstep = 4.0)
sol = integrate(HO, ImplicitMidpoint())
hams = [HarmonicOscillator.hamiltonian(0,q,p,HO.parameters) for (q,p) in zip(collect(sol.q[:]),collect(sol.p[:]))]
@show maximum(abs.((hams .- initial_hamiltonian)/initial_hamiltonian))

sol.q[:,1]
ref = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(tspan = (0,1000),tstep = 4.0))
ref.q[:,1]