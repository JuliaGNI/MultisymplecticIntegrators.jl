using MultiSymplectic
using AbstractNeuralNetworks
using Zygote
using Plots
using Profile
using NonlinearSolve

x_step = 0.01
t_step = 0.3
x_span = (0, 1)
lpde = MultiSymplectic.Wave.lpdeproblem(tstep = t_step,tspan =(0.0,t_step),xspan = x_span,xstep = x_step)

relu2(x) = max(0,x)^2
NN_width = 800
PNN = NeuralNetwork(Chain(
    Dense(2, NN_width, relu2), 
))

# scale_params = 10
# PNN.params.L1.W .*= scale_params
# PNN.params.L1.b .*= scale_params

# @benchmark Zygote.jacobian(t -> PNN([t,0.5]), 0.3)[1]
# Zygote.jacobian(x -> PNN([0.3,x]), 0.5)[1]

# PNN.params.L1.W[2] * (1-PNN([0.3,0.5])[1]^2) 



u_network = Chain(
    Dense(2, NN_width, relu2),
    Dense(NN_width, 1,identity,use_bias = false),
)

u_func = NeuralNetwork(u_network)

trial_basis = Trial_Solution_Basis(PNN,u_func,NN_width)
trial_int = TrialNN_PDE_int(trial_basis,xstep = x_step,xspan = x_span,initial_guess_method = ELM(),RT = 4,RX = 12)

trial_sol = MultiSymplectic.integrate(lpde,trial_int) 



lpde.exact_u.(0.3,collect(-4:0.01:4))
plot(trial_sol.sol.u[1], label = "t = 0.3")
plot(lpde.exact_u.(0.0,collect(-1:0.01:1)), label = "Exact")

wave_anim = @animate for (i, tt) in enumerate(0:0.1:2)
    plot(lpde.exact_u.(tt,collect(-1:0.01:1)), label = "Exact")
    
    title!("t = $tt")
    xlabel!("x")
    ylabel!("u")
end 
gif(wave_anim, fps = 5)

a = 0
b = 1
x_domain = b - a
h = 0.3

exact_u(t,x) = lpde.exact_u(t,x)

#  Trial solution function construction
psi_L(x) = (b - x) / x_domain       # left 
psi_R(x) = (x - a) / x_domain       # right
phi_B(t) = (h - t) / h   # bottom

function T1NN(t, x, dofs )
    return (b - x)  * dofs' * PNN([t,a] ) / x_domain +
        (x - a)  * dofs' * PNN([t,b]) / x_domain +
        (h - h * t)  * dofs' * PNN([0.0,x]) / h
end

function T1NN(t, x)
    return (b - x)  * W2 * tanh(W1[1]*t + W1[2]*a + bias) / x_domain +
        (x - a)  * W2 * tanh(W1[1]*t + W1[2]*b + bias) / x_domain +
        (h - h * t)  * W2 * tanh( W1[2]*x + bias) / h
end

function T2NN(t, x, dofs)
    return (b - x)   * (h - h * t)  * dofs' * PNN([0.0,a]) / x_domain / h +
        (x - a)  * (h - h * t)  * dofs' * PNN([0.0,b]) / x_domain / h
end

function C1(t, x)
    return (b - x)  * exact_u(t,a) / x_domain+
        (x - a)  * exact_u(t,b) / x_domain +
        (h - h * t)  * exact_u(0.0,x) / h
end


function C2(t, x)
    return (b - x)  * (h - h * t) * exact_u(0,a) / x_domain / h +
        (x - a)  * (h - h * t) * exact_u(0,b) / x_domain / h
end

u_trial(t,x,dofs) = dofs' * PNN([t,x]) - T1NN(t,x,dofs) + T2NN(t,x,dofs) + C1(t,x) - C2(t,x)

# x_ls = collect(a:0.01:b)
# t_ls = collect(0.0:0.01:h)
# u0 = rand(NN_width)
# u_vals = [u_trial(ti,xi,u0) for ti in t_ls, xi in x_ls]
# truth_vals = [exact_u(ti,xi) for ti in t_ls, xi in x_ls]
# u_vals - truth_vals


# v_trial(t,x,dofs) = Zygote.gradient(tt -> u_trial(tt,x,dofs),t)[1]
# w_trial(t,x,dofs) = Zygote.gradient(xx -> u_trial(t,xx,dofs),x)[1]

N_in = 1000
tx_in = rand(2,N_in)
tx_in[1,:] = tx_in[1,:] 
tx_in[2,:] = a .+ x_domain * tx_in[2,:] 


# utt(t,x,dofs) = ForwardDiff.derivative(tt -> ForwardDiff.derivative(ttt -> u_trial(ttt, x, dofs), tt), t)
# uxx(t,x,dofs) = ForwardDiff.derivative(xx -> ForwardDiff.derivative(xxx -> u_trial(t, xxx, dofs), xx), x)

function nlls!(du, u, p)
    for i in 1:N_in
        t, x = tx_in[1,i], tx_in[2,i]
        # Second derivatives
        # Wave equation: ∂²u/∂t² - c²∂²u/∂x² = 0
        # du[i] = utt(t,x,u) -  uxx(t,x,u)
        du[i] = u_trial(t,x,u) - exact_u(t,x)
    end
end

using NonlinearSolve
u0 = zeros(NN_width)
prob = NonlinearLeastSquaresProblem(
NonlinearFunction(nlls!, resid_prototype = zeros(N_in)), u0)

@time sol = solve(prob, LevenbergMarquardt(), maxiters = 100,abstol = 1e-6, reltol = 1e-6)
u_sol = sol.u

t_plot = 0:0.01:h
x_plot = 0:0.01:1
u_pred = [u_trial(t, x, sol.u) for t in t_plot, x in x_plot]
u_exact = [exact_u(t, x) for t in t_plot, x in x_plot]
error_plot = abs.(u_pred - u_exact)

# Create plots
p1 = surface(t_plot, x_plot, u_pred', title="Neural Network Solution", xlabel="t", ylabel="x_domain", zlabel="u")
p2 = surface(t_plot, x_plot, u_exact', title="Exact Solution", xlabel="x", ylabel="y", zlabel="u")
p3 = surface(t_plot, x_plot, error_plot', title="Absolute Error", xlabel="x", ylabel="y", zlabel="Error")

plot(p1, p2, p3, layout=(1,3), size=(1500, 400))


using MultiSymplectic
using AbstractNeuralNetworks
using Zygote
using Plots
using ForwardDiff
using NonlinearSolve

# Domain parameters for 2D
x_span = (0.0, 1.0)
y_span = (0.0, 1.0)
a_x, b_x = x_span[1], x_span[2]
a_y, b_y = y_span[1], y_span[2]
x_domain = b_x - a_x
y_domain = b_y - a_y
d = 2  # dimension

# Neural network for 2D input
NN_width = 800
PNN = NeuralNetwork(Chain(
    Dense(2, NN_width, tanh),  # 2D input: [x, y]
))
u0 = rand(NN_width)

# Analytical solution: u(x,y) = (1/d * (x + y))^2 + sin(1/d * (x + y))
function exact_u(x, y)
    s = (x + y) / d  # s = 1/d * ∑x_i
    return s^2 + sin(s)
end

# Compute the source term f = -∇²u analytically
function source_f(x, y)
    s = (x + y) / d  # s = 1/d * ∑x_i
    
    # First derivatives
    # ∂u/∂x = ∂u/∂s * ∂s/∂x = (2s + cos(s)) * (1/d)
    # ∂u/∂y = ∂u/∂s * ∂s/∂y = (2s + cos(s)) * (1/d)
    
    # Second derivatives
    # ∂²u/∂x² = ∂/∂x[(2s + cos(s)) * (1/d)] = (2 - sin(s)) * (1/d)²
    # ∂²u/∂y² = ∂/∂y[(2s + cos(s)) * (1/d)] = (2 - sin(s)) * (1/d)²
    
    # Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y²
    laplacian_u = 2 * (2 - sin(s)) * (1/d)^2
    
    # Source term: f = -∇²u
    return -laplacian_u
end

# Boundary shape functions for 2D
psi_L(x) = (b_x - x) / x_domain  # left boundary (x = a_x)
psi_R(x) = (x - a_x) / x_domain  # right boundary (x = b_x)
psi_B(y) = (b_y - y) / y_domain  # bottom boundary (y = a_y)
psi_T(y) = (y - a_y) / y_domain  # top boundary (y = b_y)

# Trial solution components for 2D
function T1NN(x, y, dofs)
    return psi_L(x) * dofs' * PNN([a_x, y]) +  # left boundary
           psi_R(x) * dofs' * PNN([b_x, y]) +  # right boundary
           psi_B(y) * dofs' * PNN([x, a_y]) +  # bottom boundary
           psi_T(y) * dofs' * PNN([x, b_y])    # top boundary
end

function T2NN(x, y, dofs)
    return psi_L(x) * psi_B(y) * dofs' * PNN([a_x, a_y]) +  # bottom-left corner
           psi_L(x) * psi_T(y) * dofs' * PNN([a_x, b_y]) +  # top-left corner
           psi_R(x) * psi_B(y) * dofs' * PNN([b_x, a_y]) +  # bottom-right corner
           psi_R(x) * psi_T(y) * dofs' * PNN([b_x, b_y])    # top-right corner
end

function C1(x, y)
    return psi_L(x) * exact_u(a_x, y) +  # left boundary condition
           psi_R(x) * exact_u(b_x, y) +  # right boundary condition
           psi_B(y) * exact_u(x, a_y) +  # bottom boundary condition
           psi_T(y) * exact_u(x, b_y)    # top boundary condition
end

function C2(x, y)
    return psi_L(x) * psi_B(y) * exact_u(a_x, a_y) +  # bottom-left corner
           psi_L(x) * psi_T(y) * exact_u(a_x, b_y) +  # top-left corner
           psi_R(x) * psi_B(y) * exact_u(b_x, a_y) +  # bottom-right corner
           psi_R(x) * psi_T(y) * exact_u(b_x, b_y)    # top-right corner
end

# Trial solution for 2D
u_trial(x, y, dofs) = dofs' * PNN([x, y]) - T1NN(x, y, dofs) + T2NN(x, y, dofs) + C1(x, y) - C2(x, y)

# Second derivatives for Poisson equation
uxx(x, y, dofs) = ForwardDiff.derivative(xx -> ForwardDiff.derivative(xxx -> u_trial(xxx, y, dofs), xx), x)
uyy(x, y, dofs) = ForwardDiff.derivative(yy -> ForwardDiff.derivative(yyy -> u_trial(x, yyy, dofs), yy), y)

# Collocation points for 2D domain
N_in = 2000
xy_in = rand(2, N_in)
xy_in[1, :] = a_x .+ (b_x - a_x) .* xy_in[1, :]  # Scale x to [a_x, b_x]
xy_in[2, :] = a_y .+ (b_y - a_y) .* xy_in[2, :]  # Scale y to [a_y, b_y]

# Residual function for 2D Poisson equation: ∇²u = f
function nlls!(du, u, p)
    for i in 1:N_in
        x, y = xy_in[1, i], xy_in[2, i]
        # Poisson equation: ∂²u/∂x² + ∂²u/∂y² = f(x,y)
        laplacian = uxx(x, y, u) + uyy(x, y, u)
        du[i] = laplacian - source_f(x, y)
    end
end

# Solve the nonlinear least squares problem
prob = NonlinearLeastSquaresProblem(
    NonlinearFunction(nlls!,  resid_prototype = zeros(N_in)), u0)

println("Starting optimization...")
@time sol = solve(prob,GaussNewton(), maxtime = 120)

# Visualize the solution
x_plot = 0:0.05:1
y_plot = 0:0.05:1
u_pred = [u_trial(x, y, u_sol) for x in x_plot, y in y_plot]
u_exact = [exact_u(x, y) for x in x_plot, y in y_plot]
error_plot = abs.(u_pred - u_exact)

# Create plots
p1 = surface(x_plot, y_plot, u_pred', title="Neural Network Solution", xlabel="x", ylabel="y", zlabel="u")
p2 = surface(x_plot, y_plot, u_exact', title="Exact Solution", xlabel="x", ylabel="y", zlabel="u")
p3 = surface(x_plot, y_plot, error_plot', title="Absolute Error", xlabel="x", ylabel="y", zlabel="Error")

plot(p1, p2, p3, layout=(1,3), size=(1500, 400))

# Print maximum error
max_error = maximum(error_plot)
println("\nMaximum absolute error: $max_error") 

NP = 800
@variables W1[2,1:NP] W2[1:NP] b[1:NP]
@variables t x 
nn(t,x) = W2' * tanh(W1[1,:] .*t + W1[2,:] .* x .+ b)



