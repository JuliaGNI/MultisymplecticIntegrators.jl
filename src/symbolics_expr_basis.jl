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
    function SindyPDEBasis(u_expr::Vector{Num}, P::Vector{Num},t::Num,x::Vector{Num})
        v_expr = [Symbolics.derivative(u_expr[i],t) for i in eachindex(u_expr)]
        w_expr = [Symbolics.derivative(u_expr[i],x[j]) for i in eachindex(u_expr) for j in eachindex(x)]

        ∂u_expr∂P = [Symbolics.derivative(u_expr,P[i]) for i in eachindex(P)]
        ∂v_expr∂P = [Symbolics.derivative(v_expr,P[i]) for i in eachindex(P)]
        ∂w_expr∂P = [Symbolics.derivative(w_expr[i],P[j]) for i in eachindex(w_expr) for j in eachindex(P)]
        
        ∂u∂P = [Symbolics.eval(build_function(∂u_expr∂P[i], P, t, x)) for i in eachindex(∂u_expr∂P)]
        ∂v∂P = [Symbolics.eval(build_function(∂v_expr∂P[i], P, t, x)) for i in eachindex(∂v_expr∂P)]
        ∂w∂P = [Symbolics.eval(build_function(∂w_expr∂P[i], P, t, x)) for i in eachindex(∂w_expr∂P)]

        u = eval(build_function(u_expr, P, t, x))
        v = eval(build_function(v_expr, P, t, x))
        w = [eval(build_function(w_expr[i], P, t, x)) for i in eachindex(w_expr)]
        new(u_expr, v_expr, w_expr, P, ∂u∂P, ∂v∂P, ∂w∂P, u, v, w)
    end
end