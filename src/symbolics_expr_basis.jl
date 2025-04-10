struct SindyPDEBasis
    u_expr # expression
    v_expr
    w_expr

    expr_params # parameters from the expression

    ∂u∂P # derivatives with respect to P
    ∂v∂P
    ∂w∂P

    u # callable function
    v   
    w

    Nθ::Int
    function SindyPDEBasis(u_expr::Vector{Num}, P::Symbolics.Arr{Num, 1},t::Num,x::Symbolics.Arr{Num, 1})
        v_expr = [Symbolics.derivative(u_expr[i],t) for i in eachindex(u_expr)]

        w_expr = zeros(Symbolics.Arr{Num, 1}, length(u_expr), length(x))

        for d in eachindex(u_expr)
            for dx in eachindex(x)
                    w_expr[d,dx] = Symbolics.derivative(u_expr[d],x[dx])
            end
        end

        ∂u_expr∂P = [Symbolics.derivative(u_expr,P[i]) for i in eachindex(P)]
        ∂v_expr∂P = [Symbolics.derivative(v_expr,P[i]) for i in eachindex(P)]
        ∂w_expr∂P = [Symbolics.derivative(w_expr[i],P[j]) for i in eachindex(w_expr) for j in eachindex(P)]
        
        ∂u∂P = [Symbolics.eval(build_function(∂u_expr∂P[i], P, t, x)) for i in eachindex(∂u_expr∂P)]
        ∂v∂P = [Symbolics.eval(build_function(∂v_expr∂P[i], P, t, x)) for i in eachindex(∂v_expr∂P)]
        ∂w∂P = [Symbolics.eval(build_function(∂w_expr∂P[i], P, t, x)) for i in eachindex(∂w_expr∂P)]

        u = eval(build_function(u_expr, P, t, x))
        v = eval(build_function(v_expr, P, t, x))
        w = zeros(Function, length(u_expr), length(x))

        for d in eachindex(u_expr)
            for dx in eachindex(x)
                w[d,dx] = eval(build_function(w_expr[d,dx], P, t, x))
            end
        end

        Nθ = length(P)
        new(u_expr, v_expr, w_expr, P, ∂u∂P, ∂v∂P, ∂w∂P, u, v, w, Nθ)
    end
end