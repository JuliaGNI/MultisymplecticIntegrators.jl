using MultiSymplectic
using AbstractNeuralNetworks
lpde = MultiSymplectic.LinearTransport.lpdeproblem(tstep = 0.3,tspan =(0.0,1.5),xspan = (-4.,-1.))

u_network = Chain(
    Dense(2, 100, tanh),
    Dense(100, 100, tanh),
    Dense(100, 1, tanh)
)
PNN = NeuralNetwork(u_network)
trial_basis = Trial_Solution_Basis(PNN,lpde,:TFC)
trial_basis.u(0.1, 0.25, 0.0, PNN.params)
using Plots

h = 0.3
xspan = (-4.0, -1.0)
a = xspan[1]
b = xspan[2]
x_length = b - a

exact_u = MultiSymplectic.LinearTransport.exact_u

T₁NN(t,x,tn,params) = ((tn + h -t)/h)*PNN([tn,x],params)[1] + ((b-x)/(b-a))*PNN([t,a],params)[1] + ((x-a)/(b-a))*PNN([t,b],params)[1]
T₂NN(t,x,tn,params) = ((tn + h -t)*(b-x)/(x_length * h))*PNN([tn,a],params)[1] + ((tn + h -t)*(x-a)/(x_length * h))*PNN([tn,b],params)[1]
C₁(t,x,tn) = ((tn + h -t)/h)*exact_u(0,x) + ((b-x)/(b-a))*exact_u(xspan[1],t) + ((x-a)/(b-a))*exact_u(xspan[2],t)
C₂(t,x,tn) = ((tn + h -t)*(b-x)/(x_length * h))*exact_u(xspan[1],0) + ((tn + h -t)*(x-a)/(x_length * h))*exact_u(xspan[2],t)

u_trial(t,x,tn,params) = PNN([t,x],params)[1]- T₁NN(t,x,tn,params) + T₂NN(t,x,tn,params) + C₁(t,x,tn) - C₂(t,x,tn)
v_trial(t,x,tn,params) = Zygote.gradient(tt -> u_trial(tt,x,tn,params),t)[1]
w_trial(t,x,tn,params) = Zygote.gradient(xx -> u_trial(t,xx,tn,params),x)[1]



analytic_sol = [MultiSymplectic.LinearTransport.exact_u(t,x) for t in t_vals, x in x_vals]
Z = [u_trial(t, x, 0.0, PNN.params) for t in t_vals, x in x_vals]
heatmap(x_vals, t_vals, analytic_sol .- Z, xlabel="x", ylabel="t")


t_vals = range(0.0, 0.3, length=50)
x_vals = range(-4.0, -1.0, length=100)

# 权函数
phi_L(t, tn, h) = (tn + h - t) / h       # 左面权 (t=tn 时 =1)
phi_R(t, tn, h) = (t - tn) / h           # 右面权 (t=tn+h 时 =1)
phi_B(x, a, b)  = (b - x) / (b - a)     # 下边权 (x=a 时 =1)

# 面项 (T1NN) —— 只保留左/右/下三面
function T1NN(t, x, tn, PNN)
    return phi_L(t,tn,h) * PNN([tn,     x], PNN.params)[1] +
           phi_R(t,tn,h) * PNN([tn + h, x], PNN.params)[1] +
           phi_B(x,a,b)  * PNN([t,      a], PNN.params)[1]
end

# 角项 (T2NN) —— 只保留左下角和右下角
function T2NN(t, x, tn, PNN)
    return phi_L(t,tn,h) * phi_B(x,a,b) * PNN([tn,     a], PNN.params)[1] +
           phi_R(t,tn,h) * phi_B(x,a,b) * PNN([tn + h, a], PNN.params)[1]
end

# C1, C2 —— 用 exact_u 构造的已知边界补偿项
function C1(t, x, tn)
    return phi_L(t,tn,h) * exact_u(tn,     x) +
           phi_R(t,tn,h) * exact_u(tn + h, x) +
           phi_B(x,a,b)  * exact_u(t,      a)
end

function C2(t, x)
    return phi_L(t,tn,h) * phi_B(x,a,b) * exact_u(tn,     a) +
           phi_R(t,tn,h) * phi_B(x,a,b) * exact_u(tn + h, a)
end

# 最终受约束 trial 解
function u_trial(t, x, params, PNN)
    g_val  = PNN([t,x], params)[1]
    T1_val = T1NN(t, x, tn, PNN)
    T2_val = T2NN(t, x, tn, PNN)
    C1_val = C1(t, x, tn)
    C2_val = C2(t, x)
    return g_val - T1_val + T2_val + C1_val - C2_val
end

Z = [u_trial(t, x, PNN.params, PNN) for t in t_vals, x in x_vals]
analytic_sol = [MultiSymplectic.LinearTransport.exact_u(t,x) for t in t_vals, x in x_vals]
Z .- analytic_sol

heatmap(x_vals, t_vals, Z .- analytic_sol, xlabel="x", ylabel="t")

using Plots

# ==== 区间常量 ====
t0 = 0.0           # 下边界时间
t1 = 0.3           # 上边界时间 (自由)
a  = -4.0          # 左边界 x=a
b  = -1.0          # 右边界 x=b


# ==== 网格采样 ====
t_vals = range(0.0, 0.3, length=50)
x_vals = range(a, b, length=100)

Z = [u_trial(t, x, 0.0, 0.3, a, b, PNN.params, PNN) for t in t_vals, x in x_vals]
analytic_sol = [exact_u(t,x) for t in t_vals, x in x_vals]

Δ = Z .- analytic_sol

heatmap(x_vals, t_vals, Δ,
    xlabel="x", ylabel="t", title="u_trial - exact_u",
    colorbar_title="误差")

# ==== 检查三条边误差 ====
println("检查左边 x=a 的误差:")
for t in t_vals[1:10:end]
    println("t=$t, 误差=", u_trial(t,a,t0,t1,a,b,PNN_model.params,PNN) - exact_u(t,a))
end

println("检查右边 x=b 的误差:")
for t in t_vals[1:10:end]
    println("t=$t, 误差=", u_trial(t,b,t0,t1,a,b,PNN_model.params,PNN) - exact_u(t,b))
end

println("检查下边 t=t0 的误差:")
