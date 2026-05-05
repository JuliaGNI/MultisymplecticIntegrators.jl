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

    function Trial_Solution_Basis(S::Int, σ::AF, exact_u_expr,x_span) where {AF}
        a,b = x_span[1], x_span[2]
        u_expr, (t_sym, x_sym, tn_sym, h_sym), (W2_s, W1_s, b1_s, previous_W2_s, previous_W1_s, previous_bias1_s) = generate_symbolic_u_trial(S, σ, exact_u_expr,a=a, b=b)
        # 1. 对时间 t 求导 (v_trial)
        v_expr = Symbolics.derivative(u_expr, t_sym)

        # 2. 对空间 x 求导 (w_trial)
        w_expr = Symbolics.derivative(u_expr, x_sym)

        # 3. 对参数 p 求导 (∂u∂p)
        # 把所有训练参数打包成一个向量
        params_flat = [vec(W2_s); vec(W1_s); vec(b1_s)]
        du_dp_expr = Symbolics.jacobian([u_expr], params_flat) 
        dv_dp_expr = Symbolics.jacobian([v_expr], params_flat)
        dw_dp_expr = Symbolics.jacobian([w_expr], params_flat)

        # 4. 编译为 Julia 函数
        # build_function 会生成非常高效的、包含展开循环的代码
        # target=:function 生成普通函数，target=:inplace 可以生成不分配内存的版本
        u_func = build_function(u_expr, t_sym, x_sym, tn_sym, h_sym, params_flat, previous_W2_s, previous_W1_s, previous_bias1_s, expression=Val{false})
        v_func = build_function(v_expr, t_sym, x_sym, tn_sym, h_sym, params_flat, previous_W2_s, previous_W1_s, previous_bias1_s, expression=Val{false})
        w_func = build_function(w_expr, t_sym, x_sym, tn_sym, h_sym, params_flat, previous_W2_s, previous_W1_s, previous_bias1_s, expression=Val{false})

        ∂u∂p_func = build_function(du_dp_expr, t_sym, x_sym, tn_sym, h_sym, params_flat, previous_W2_s, previous_W1_s, previous_bias1_s, expression=Val{false})[1]
        ∂v∂p_func = build_function(dv_dp_expr, t_sym, x_sym, tn_sym, h_sym, params_flat, previous_W2_s, previous_W1_s, previous_bias1_s, expression=Val{false})[1]
        ∂w∂p_func = build_function(dw_dp_expr, t_sym, x_sym, tn_sym, h_sym, params_flat, previous_W2_s, previous_W1_s, previous_bias1_s, expression=Val{false})[1]


        new{AF, typeof(u_func), typeof(v_func), typeof(w_func), 
        typeof(∂u∂p_func), typeof(∂v∂p_func), typeof(∂w∂p_func)}(σ, S, 4 * S, 
        u_func, v_func, w_func, ∂u∂p_func, ∂v∂p_func, ∂w∂p_func)#, BNN, PNN
    end
end

function generate_symbolic_u_trial(S::Int, activation_fn, exact_u_sym_fn; a=0.0, b=1.0)
    # 1. 定义符号变量
    @variables t x tn h 
    @variables W2[1:S] W1[1:S, 1:2] bias1[1:S]
    # 如果 C1 中涉及上一时刻的权重，也需要定义为符号常数（或者直接传入数值）
    @variables previous_W2[1:S] previous_W1[1:S, 1:2] previous_bias1[1:S]

    # 2. 定义符号 NN
    # 映射激活函数 (如果是自定义函数，需要先注册，例如 @register_symbolic my_act(x))
    # 假设 activation_fn 是可以直接处理 Symbolics.Num 的函数，如 σ(x) = 1/(1+exp(-x))
    function sym_NN(_t, _x, _W2, _W1, _b1)
        return sum(_W2[i] * activation_fn(_W1[i,1]*_t + _W1[i,2]*_x + _b1[i]) for i in 1:S)
    end

    x_domain = b - a
    
    # 3. 构造辅助项
    # T1NN_manual (使用符号 h)
    t1_nn = (b - x) / x_domain * sym_NN(t, a, W2, W1, bias1) +
            (x - a) / x_domain * sym_NN(t, b, W2, W1, bias1) +
            (h - h * t) / h * sym_NN(0.0, x, W2, W1, bias1)

    # T2NN_manual (使用符号 h)
    t2_nn = (b - x) / x_domain * (h - h * t) / h * sym_NN(0.0, a, W2, W1, bias1) +
            (x - a) / x_domain * (h - h * t) / h * sym_NN(0.0, b, W2, W1, bias1)

    # C1 的特殊处理：使用 IfElse.ifelse 代替普通 if
    # 如果 tn == 0, 使用 exact_u，否则使用上一时刻的 NN (TrialNN 逻辑)
    prev_step_term = ifelse(tn == 0.0, 
                    exact_u_sym_fn(0.0, x), 
                    sym_NN(1.0, x, previous_W2, previous_W1, previous_bias1))

    c1 = (b - x) * exact_u_sym_fn(tn + h*t, a) / x_domain +
         (x - a) * exact_u_sym_fn(tn + h*t, b) / x_domain +
         (h - h * t) * prev_step_term / h

    # C2 (使用符号 tn 和 h)
    c2 = (b - x) * (h - h * t) * exact_u_sym_fn(tn, a) / x_domain / h +
         (x - a) * (h - h * t) * exact_u_sym_fn(tn, b) / x_domain / h

    # 4. 组合得到 u_trial 符号表达式
    expr_u = sym_NN(t, x, W2, W1, bias1) - t1_nn + t2_nn + c1 - c2

    # 返回时，包含 tn 和 h 在自变量列表中
    return expr_u, (t, x, tn, h), (W2, W1, bias1, previous_W2, previous_W1, previous_bias1)
end