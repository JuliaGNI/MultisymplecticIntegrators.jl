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

    function Trial_Solution_Basis(S::Int, σ::AF,x_span) where {AF}
        a,b = x_span[1], x_span[2]
        u_expr, (t_sym, x_sym, h_sym), (W2_s, W1_s, b1_s) = generate_symbolic_u_trial(S, σ,a=a, b=b) # previous_W2_s, previous_W1_s, previous_bias1_s,
        # 1. Differentiate with respect to time t (v_trial)
        v_expr = Symbolics.derivative(u_expr, t_sym)

        # 2. Differentiate with respect to space x (w_trial)
        w_expr = Symbolics.derivative(u_expr, x_sym)

        # 3. Differentiate with respect to parameters p (∂u∂p)
        # Pack all trainable parameters into a single vector
        params_flat = [vec(W2_s); vec(W1_s); vec(b1_s)]
        du_dp_expr = Symbolics.jacobian([u_expr], params_flat) 
        dv_dp_expr = Symbolics.jacobian([v_expr], params_flat)
        dw_dp_expr = Symbolics.jacobian([w_expr], params_flat)

        # 4. Compile to Julia functions
        # build_function generates very efficient code with unrolled loops
        # target=:function creates a normal function, target=:inplace can generate an allocation-free version
        u_func = build_function(u_expr, t_sym, x_sym, h_sym, params_flat, expression=Val{false}) 
        v_func = build_function(v_expr, t_sym, x_sym, h_sym, params_flat, expression=Val{false})
        w_func = build_function(w_expr, t_sym, x_sym, h_sym, params_flat, expression=Val{false})

        ∂u∂p_func = build_function(du_dp_expr, t_sym, x_sym, h_sym, params_flat, expression=Val{false})[1]
        ∂v∂p_func = build_function(dv_dp_expr, t_sym, x_sym, h_sym, params_flat, expression=Val{false})[1]
        ∂w∂p_func = build_function(dw_dp_expr, t_sym, x_sym, h_sym, params_flat, expression=Val{false})[1]


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

function generate_symbolic_u_trial(S::Int, activation_fn; a=0.0, b=1.0)
    # 1. Define symbolic variables
    @variables t x h 
    @variables W2[1:S] W1[1:S, 1:2] bias1[1:S]
    # If C1 depends on previous-step weights, define them as symbolic constants (or pass values directly)
    # @variables previous_W2[1:S] previous_W1[1:S, 1:2] previous_bias1[1:S]

    # 2. Define the symbolic neural network
    # Map the activation function (for custom functions, register it first, e.g. @register_symbolic my_act(x))
    function sym_NN(_t, _x, _W2, _W1, _b1)
        return sum(_W2[i] * activation_fn(_W1[i,1]*_t + _W1[i,2]*_x + _b1[i]) for i in 1:S)
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
