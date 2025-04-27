
struct LPDESystem
    L
    D
    t 
    x
    u # alias for q
    v # alias for ut
    w # alias for ux
    params
    equations
    functions
    function LPDESystem(L,t,x,U,V,W,params = NamedTuple();simplify = true, scalarize = true)

        DX = length(x)
        D = length(U)

        Ls = scalarize ? Symbolics.scalarize(L) : L
        Ls = simplify ? Symbolics.simplify(Ls) : Ls

        ∂L∂U_expr = [Symbolics.derivative(Ls, U[i]) for i in eachindex(U)]
        ∂L∂V_expr = [Symbolics.derivative(Ls, V[i]) for i in eachindex(V)]
        ∂L∂W_expr = [Symbolics.derivative(Ls, W[i]) for i in eachindex(W)]

        # ∂L∂W_expr = zeros(Num,D,DX)
        # for d in 1:D
        #     for dx in 1:DX
        #         ∂L∂W_expr[d,dx] = Symbolics.derivative(Ls, W[d,dx])
        #     end
        # end

        equs = (
            L = Ls,
            ∂L∂U = ∂L∂U_expr,
            ∂L∂V = ∂L∂V_expr,
            ∂L∂W = ∂L∂W_expr,
        ) # set of expressions
        sparams = symbolize(params)

        ∂L∂U = [Symbolics.eval(substitute_parameters(Symbolics.build_function(∂L∂U_expr[i], U, V, W,sparams...; nanmath = false),sparams)) for i in eachindex(∂L∂U_expr)]
        ∂L∂V = [Symbolics.eval(substitute_parameters(Symbolics.build_function(∂L∂V_expr[i], U, V, W,sparams...; nanmath = false),sparams)) for i in eachindex(∂L∂V_expr)]
        ∂L∂W = [Symbolics.eval(substitute_parameters(Symbolics.build_function(∂L∂W_expr[i], U, V, W,sparams...; nanmath = false),sparams)) for i in eachindex(∂L∂W_expr)]

        # ∂L∂W = Array{Function}(undef,D,DX)
        # for d in 1:D
        #     for dx in 1:DX
        #         ∂L∂W[d,dx] = Symbolics.eval(substitute_parameters(Symbolics.build_function(∂L∂W_expr[d,dx], U, V, W,sparams...; nanmath = false),sparams))
        #     end
        # end

        functions = (
            ∂L∂U = ∂L∂U,
            ∂L∂V = ∂L∂V,
            ∂L∂W = ∂L∂W,
        ) # set of callable functions

        return new(Ls,D, t, x, U,V,W, params, equs, functions)
    end
end

