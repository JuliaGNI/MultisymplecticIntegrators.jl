using MultiSymplectic
using AbstractNeuralNetworks
using Zygote
using Plots
using Profile

x_step = 1.0
t_step = 1.0
x_span = (0.0,1.0)
lpde = MultiSymplectic.LinearTransport.lpdeproblem(tstep = t_step,tspan =(0.0,t_step),xspan = x_span,xstep = x_step)

NP = 50
basis_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh,),
    Dense(100, 100, tanh,),
    Dense(100, NP, tanh),
)

PNN_basis = NeuralNetwork(basis_network)

u_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, NP, tanh),
    Dense(NP, 1,identity,use_bias = false),
)

u_func = NeuralNetwork(u_network)
trial_basis = Trial_Solution_Basis(PNN_basis,u_func,NP)
trial_int = TrialNN_PDE_int(trial_basis,xstep = x_step,xspan = x_span)

trial_sol = MultiSymplectic.integrate(lpde,trial_int)
lpde.exact_u.(0.3,collect(-4:0.01:-1))
plot(trial_sol.sol.u[1], label = "t = 0.3")
plot(lpde.exact_u.(0.0,collect(-4:0.01:-1)), label = "Exact")


NN_width = 250
PNN = NeuralNetwork(Chain(
    Dense(2, NN_width, tanh),
))
u0 = rand(NN_width)
u0' * PNN([0.0,-4.0])

a = 0
b = 1
x_domain = b - a
h = 1.0 

exact_u(t,x) = lpde.exact_u(t,x)

#  Trial solution function construction
psi_L(x) = (b - x) / x_domain       # left 
psi_R(x) = (x - a) / x_domain       # right
phi_B(t) = (h - t) / h   # bottom

function T1NN(t, x, dofs )
    return (b - x)  * dofs' * PNN([t,a] ) / x_domain +
        (x - a)  * dofs' * PNN([t,b]) / x_domain +
        (h - t)  * dofs' * PNN([0.0,x]) / h
end

function T2NN(t, x, dofs)
    return (b - x)   * (h - t)  * dofs' * PNN([0.0,a]) / x_domain / h +
        (x - a)  * (h - t)  * dofs' * PNN([0.0,b]) / x_domain / h
end

function C1(t, x)
    return (b - x)  * exact_u(t,a) / x_domain+
        (x - a)  * exact_u(t,b) / x_domain +
        (h - t)  * exact_u(0.0,x) / h
end

function C2(t, x)
    return (b - x)  * (h - t) * exact_u(0,a) / x_domain / h +
        (x - a)  * (h - t) * exact_u(0,b) / x_domain / h
end


u_trial(t,x,dofs) = dofs' * PNN([t,x]) - T1NN(t,x,dofs) + T2NN(t,x,dofs) + C1(t,x) - C2(t,x)

x_ls = collect(a:0.01:b)
t_ls = collect(0.0:0.01:h)

u_vals = [u_trial(ti,xi,u0) for ti in t_ls, xi in x_ls]
truth_vals = [exact_u(ti,xi) for ti in t_ls, xi in x_ls]
u_vals - truth_vals


v_trial(t,x,dofs) = Zygote.gradient(tt -> u_trial(tt,x,dofs),t)[1]
w_trial(t,x,dofs) = Zygote.gradient(xx -> u_trial(t,xx,dofs),x)[1]

N_in = 600
tx_in = rand(2,N_in)

function nlls!(du, u, p)
    for i in 1:N_in
        du[i] = v_trial(tx_in[1,i], tx_in[2,i],  u)  + w_trial(tx_in[1,i], tx_in[2,i],u)
    end
end


prob = NonlinearLeastSquaresProblem(
NonlinearFunction(nlls!, resid_prototype = zeros(N_in)), u0)

@time u_sol = solve(prob,maxtime = 60,abstol = 1e-12, reltol = 1e-12).u
(x == cache(int).x && eltype(x) == eltype(cache(int).x)) ? x[:] = u_sol : nothing
println("Time for initial guess: ", time() - t1)
print("initial guess parameters: ", x, "\n")
C.done_initial_guess[1] = 1
