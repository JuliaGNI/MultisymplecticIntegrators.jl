struct Trial_Solution_Basis{AF, UFT, VFT, WFT, UPFT, VPFT, WPFT} <: AbstractPDEBasis
    activation_function::AF
    S::Int
    NP::Int

    u_func::UFT
    v_func::VFT
    w_func::WFT

    ∂u∂p_func::UPFT
    ∂v∂p_func::VPFT
    ∂w∂p_func::WPFT

    function Trial_Solution_Basis(S::Int, σ::AF, x_span) where {AF}
        a, b = x_span[1], x_span[2]
        funcs = build_trial_solution_functions(S, σ, a, b)
        u_func, v_func, w_func, ∂u∂p_func, ∂v∂p_func, ∂w∂p_func = funcs

        new{AF, typeof(u_func), typeof(v_func), typeof(w_func),
            typeof(∂u∂p_func), typeof(∂v∂p_func), typeof(∂w∂p_func)}(σ, S, 4 * S,
            u_func, v_func, w_func, ∂u∂p_func, ∂v∂p_func, ∂w∂p_func)#, BNN, PNN
    end
end

function Base.show(io::IO, basis::Trial_Solution_Basis)
    print(io, "\n Trial Solution Neural Network Basis with:\n")
    print(io, "   Hidden units S: $(basis.S) \n")
    print(io, "   Number of optimized parameters NP: $(basis.NP) \n")
    print(io, "   Activation function: $(basis.activation_function) \n")
end

function generate_symbolic_u_trial(S::Int, activation_fn; a = 0.0, b = 1.0)
    # 1. Define symbolic variables
    @variables t x h
    @variables W2[1:S] W1[1:S, 1:2] bias1[1:S]
    # If C1 depends on previous-step weights, define them as symbolic constants (or pass values directly)
    # @variables previous_W2[1:S] previous_W1[1:S, 1:2] previous_bias1[1:S]

    # 2. Define the symbolic neural network
    # Map the activation function (for custom functions, register it first, e.g. @register_symbolic my_act(x))
    function sym_NN(_t, _x, _W2, _W1, _b1)
        return sum(_W2[i] * activation_fn(_W1[i, 1]*_t + _W1[i, 2]*_x + _b1[i])
        for i in 1:S)
    end

    x_domain = b - a

    # 3. Build auxiliary terms
    # T1NN_manual uses symbolic h
    t1_nn = (b - x) / x_domain * sym_NN(t, a, W2, W1, bias1) +
            (x - a) / x_domain * sym_NN(t, b, W2, W1, bias1) +
            (h - h * t) / h * sym_NN(0.0, x, W2, W1, bias1)

    # T2NN_manual uses symbolic h
    t2_nn = (b - x) / x_domain * (h - h * t) / h * sym_NN(0.0, a, W2, W1, bias1) +
            (x - a) / x_domain * (h - h * t) / h * sym_NN(0.0, b, W2, W1, bias1)

    # # Special handling for C1: use IfElse.ifelse instead of a plain if
    # # If tn == 0, use exact_u; otherwise use previous-step NN (TrialNN logic)
    # prev_step_term = ifelse(tn == 0.0, 
    #                 exact_u_sym_fn(0.0, x), 
    #                 sym_NN(1.0, x, previous_W2, previous_W1, previous_bias1))

    # c1 = (b - x) * exact_u_sym_fn(tn + h*t, a) / x_domain +
    #      (x - a) * exact_u_sym_fn(tn + h*t, b) / x_domain +
    #      (h - h * t) * prev_step_term / h

    # # C2 uses symbolic tn and h
    # c2 = (b - x) * (h - h * t) * exact_u_sym_fn(tn, a) / x_domain / h +
    #      (x - a) * (h - h * t) * exact_u_sym_fn(tn, b) / x_domain / h

    # 4. Combine terms to obtain the symbolic u_trial expression
    expr_u = sym_NN(t, x, W2, W1, bias1) - t1_nn + t2_nn # + c1 - c2

    # Return with tn and h included in the argument list
    return expr_u, (t, x, h), (W2, W1, bias1)# , previous_W2, previous_W1, previous_bias1
end

@inline activation_prime(::typeof(tanh), z) = one(z) - tanh(z)^2
@inline activation_second(::typeof(tanh), z) = -2 * tanh(z) * activation_prime(tanh, z)

@inline activation_prime(σ, z) = ForwardDiff.derivative(σ, z)
@inline activation_second(σ, z) = ForwardDiff.derivative(zz -> activation_prime(σ, zz), z)

function build_trial_solution_functions(S::Int, σ, a, b)
    u_func = (t, x, h, params) -> trial_solution_values(S, σ, a, b, t, x, params)[1]
    v_func = (t, x, h, params) -> trial_solution_values(S, σ, a, b, t, x, params)[2]
    w_func = (t, x, h, params) -> trial_solution_values(S, σ, a, b, t, x, params)[3]

    function trial_param_derivatives(t, x, params)
        du = zeros(typeof(t + x + first(params)), 4S)
        dv = similar(du)
        dw = similar(du)
        trial_solution_derivatives!(du, dv, dw, S, σ, a, b, t, x, params)
        return du, dv, dw
    end

    ∂u∂p_func = (t, x, h, params) -> trial_param_derivatives(t, x, params)[1]
    ∂v∂p_func = (t, x, h, params) -> trial_param_derivatives(t, x, params)[2]
    ∂w∂p_func = (t, x, h, params) -> trial_param_derivatives(t, x, params)[3]

    return u_func, v_func, w_func, ∂u∂p_func, ∂v∂p_func, ∂w∂p_func
end

function trial_solution_values(S::Int, σ, a, b, t, x, params)
    x_domain = b - a
    W2 = @view params[1:S]
    W1t = @view params[(S + 1):(2 * S)]
    W1x = @view params[(2 * S + 1):(3 * S)]
    bias1 = @view params[(3 * S + 1):(4 * S)]

    L = (b - x) / x_domain
    R = (x - a) / x_domain
    B = one(t) - t

    u = zero(t + x + first(params))
    v = zero(u)
    w = zero(u)

    for i in 1:S
        α = W1t[i]
        β = W1x[i]
        γ = bias1[i]

        ztx = α * t + β * x + γ
        zta = α * t + β * a + γ
        ztb = α * t + β * b + γ
        z0x = β * x + γ
        z0a = β * a + γ
        z0b = β * b + γ

        Stx = σ(ztx)
        Sta = σ(zta)
        Stb = σ(ztb)
        S0x = σ(z0x)
        S0a = σ(z0a)
        S0b = σ(z0b)

        Dtx = activation_prime(σ, ztx)
        Dta = activation_prime(σ, zta)
        Dtb = activation_prime(σ, ztb)
        D0x = activation_prime(σ, z0x)

        φ = Stx - L * Sta - R * Stb - B * S0x + L * B * S0a + R * B * S0b
        φt = α * (Dtx - L * Dta - R * Dtb) + S0x - L * S0a - R * S0b
        φx = β * Dtx + (Sta - Stb) / x_domain - B * β * D0x + B * (-S0a + S0b) / x_domain

        u += W2[i] * φ
        v += W2[i] * φt
        w += W2[i] * φx
    end

    return u, v, w
end

function trial_solution_derivatives!(du, dv, dw, S::Int, σ, a, b, t, x, params)
    trial_solution_values_and_derivatives!(du, dv, dw, S, σ, a, b, t, x, params)
    return du, dv, dw
end

function trial_solution_values_and_derivatives!(du, dv, dw, S::Int, σ, a, b, t, x, params)
    x_domain = b - a
    W2 = @view params[1:S]
    W1t = @view params[(S + 1):(2 * S)]
    W1x = @view params[(2 * S + 1):(3 * S)]
    bias1 = @view params[(3 * S + 1):(4 * S)]

    L = (b - x) / x_domain
    R = (x - a) / x_domain
    B = one(t) - t
    u = zero(t + x + first(params))
    v = zero(u)
    w = zero(u)

    for i in 1:S
        α = W1t[i]
        β = W1x[i]
        γ = bias1[i]

        ztx = α * t + β * x + γ
        zta = α * t + β * a + γ
        ztb = α * t + β * b + γ
        z0x = β * x + γ
        z0a = β * a + γ
        z0b = β * b + γ

        Stx = σ(ztx)
        Sta = σ(zta)
        Stb = σ(ztb)
        S0x = σ(z0x)
        S0a = σ(z0a)
        S0b = σ(z0b)

        Dtx = activation_prime(σ, ztx)
        Dta = activation_prime(σ, zta)
        Dtb = activation_prime(σ, ztb)
        D0x = activation_prime(σ, z0x)
        D0a = activation_prime(σ, z0a)
        D0b = activation_prime(σ, z0b)

        Htx = activation_second(σ, ztx)
        Hta = activation_second(σ, zta)
        Htb = activation_second(σ, ztb)
        H0x = activation_second(σ, z0x)

        φ = Stx - L * Sta - R * Stb - B * S0x + L * B * S0a + R * B * S0b
        φt = α * (Dtx - L * Dta - R * Dtb) + S0x - L * S0a - R * S0b
        φx = β * Dtx + (Sta - Stb) / x_domain - B * β * D0x + B * (-S0a + S0b) / x_domain
        u += W2[i] * φ
        v += W2[i] * φt
        w += W2[i] * φx

        φα = t * (Dtx - L * Dta - R * Dtb)
        φβ = x * Dtx - L * a * Dta - R * b * Dtb - B * x * D0x + L * B * a * D0a +
             R * B * b * D0b
        φγ = Dtx - L * Dta - R * Dtb - B * D0x + L * B * D0a + R * B * D0b

        φtα = (Dtx - L * Dta - R * Dtb) + α * t * (Htx - L * Hta - R * Htb)
        φtβ = α * (x * Htx - L * a * Hta - R * b * Htb) + x * D0x - L * a * D0a -
              R * b * D0b
        φtγ = α * (Htx - L * Hta - R * Htb) + D0x - L * D0a - R * D0b

        φxα = β * t * Htx + t * (Dta - Dtb) / x_domain
        φxβ = Dtx + β * x * Htx + (a * Dta - b * Dtb) / x_domain -
              B * (D0x + β * x * H0x) + B * (-a * D0a + b * D0b) / x_domain
        φxγ = β * Htx + (Dta - Dtb) / x_domain - B * β * H0x + B * (-D0a + D0b) / x_domain

        du[i] = φ
        du[S + i] = W2[i] * φα
        du[2 * S + i] = W2[i] * φβ
        du[3 * S + i] = W2[i] * φγ

        dv[i] = φt
        dv[S + i] = W2[i] * φtα
        dv[2 * S + i] = W2[i] * φtβ
        dv[3 * S + i] = W2[i] * φtγ

        dw[i] = φx
        dw[S + i] = W2[i] * φxα
        dw[2 * S + i] = W2[i] * φxβ
        dw[3 * S + i] = W2[i] * φxγ
    end

    return u, v, w
end
