

struct NetworkPDEBasis_Symbolic{OMT,AF,UF,VF,WF,UPT,VPT,WPT} <: AbstractPDEBasis
    S::Int
    activation_function::AF

    u::UF
    v::VF
    w::WF

    ∂u∂P::UPT # derivatives with respect to P
    ∂v∂P::VPT
    ∂w∂P::WPT

    NP::Int
    optim_mode::OMT # :Partially or :Fully, whether to solve only last layer parameters or all the parameters in the neural network
    function NetworkPDEBasis_Symbolic(S, activation_function::AF,optim_mode::OMT = :Partially; XT_dim = 2) where {OMT,AF}
        @variables t_sym x_sym
        @variables W2[1:S] W1[1:S, 1:XT_dim] bias1[1:S]
        # 2. Define the symbolic neural network
        # Map the activation function (for custom functions, register it first, e.g. @register_symbolic my_act(x))

        function sym_NN(_t, _x, _W2, _W1, _b1)
            return sum(_W2[i] * activation_function(_W1[i,1]*_t + _W1[i,2]*_x + _b1[i]) for i in 1:S)
        end

        u_expr = sym_NN(t_sym, x_sym, W2, W1, bias1)
        # 1. Differentiate with respect to time t (v_trial)
        v_expr = Symbolics.derivative(u_expr, t_sym)

        # 2. Differentiate with respect to space x (w_trial)
        w_expr = Symbolics.derivative(u_expr, x_sym)

        # 3. Differentiate with respect to parameters p (∂u∂p)
        # Pack all trainable parameters into a single vector
        params_flat = [vec(W1); vec(bias1); vec(W2)]
        du_dp_expr = Symbolics.jacobian([u_expr], params_flat)
        dv_dp_expr = Symbolics.jacobian([v_expr], params_flat)
        dw_dp_expr = Symbolics.jacobian([w_expr], params_flat)

        # 4. Compile to Julia functions
        # build_function generates very efficient code with unrolled loops
        u_func = build_function(u_expr, t_sym, x_sym, params_flat, expression=Val{false})
        v_func = build_function(v_expr, t_sym, x_sym, params_flat, expression=Val{false})
        w_func = build_function(w_expr, t_sym, x_sym, params_flat, expression=Val{false})

        ∂u∂P_func = build_function(du_dp_expr, t_sym, x_sym, params_flat, expression=Val{false})[1]
        ∂v∂P_func = build_function(dv_dp_expr, t_sym, x_sym, params_flat, expression=Val{false})[1]
        ∂w∂P_func = build_function(dw_dp_expr, t_sym, x_sym, params_flat, expression=Val{false})[1]

        if optim_mode == :Fully
            NP = (XT_dim + 2) * S # all parameters in the two-layer network
        elseif optim_mode == :Partially
            NP = S
        else
            error("Invalid optim_mode. Use :Partially or :Fully.")
        end

        return new{OMT,AF,typeof(u_func),typeof(v_func),typeof(w_func),typeof(∂u∂P_func),typeof(∂v∂P_func),typeof(∂w∂P_func)}(S, activation_function,
            u_func, v_func, w_func,
            ∂u∂P_func, ∂v∂P_func, ∂w∂P_func,
            NP,optim_mode)
    end
end

function Base.show(io::IO, basis::NetworkPDEBasis_Symbolic)
    print(io, "\n Symbolics.jl Neural Network PDE Basis with:\n")
    print(io, "   Hidden units S: $(basis.S) \n")
    print(io, "   Number of optimized parameters NP: $(basis.NP) \n")
    print(io, "   Optim mode: $(basis.optim_mode) \n")
    print(io, "   Activation function: $(basis.activation_function) \n")
end
