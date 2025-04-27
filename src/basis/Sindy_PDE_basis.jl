struct SindyPDEBasis <: AbstractPDEBasis
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

    NP::Int
    P_sizes::Vector{Int}

    function SindyPDEBasis(u_expr::Vector{Num}, P::Vector{Symbolics.Arr{Num, 1}},t::Num,x::Vector{Num})
        P_sizes = map(length,P)
        NP = sum(P_sizes)
        D = length(u_expr)
        DX = length(x)
        
        v_expr = [Symbolics.derivative(u_expr[i],t) for i in 1:D]
        w_expr = [Symbolics.derivative(u_expr[i],x[j]) for i in 1:D, j in 1:DX]

        # Compute the derivatives of u and v with respect to P
        ∂u∂P = []
        ∂v∂P = []
        ∂w∂P = []
        for d in 1:D
            ∂u_expr∂P = zeros(Num,P_sizes[d])
            ∂v_expr∂P = zeros(Num,P_sizes[d])
            ∂w_expr∂P = zeros(Num,P_sizes[d])
            for i in 1:P_sizes[d]
                ∂u_expr∂P[i] = Symbolics.derivative(u_expr[d],P[d][i])
                ∂v_expr∂P[i] = Symbolics.derivative(v_expr[d],P[d][i])
                ∂w_expr∂P[i] = Symbolics.derivative(u_expr[d],P[d][i])
            end
            ∂u∂P = [Symbolics.eval(Symbolics.build_function(∂u_expr∂P[i],P, t, x)) for i in 1:P_sizes[d]]
            ∂v∂P = [Symbolics.eval(Symbolics.build_function(∂v_expr∂P[i],P, t, x)) for i in 1:P_sizes[d]]
            ∂w∂P = [Symbolics.eval(Symbolics.build_function(∂w_expr∂P[i],P, t, x)) for i in 1:P_sizes[d]]

            # push!(∂u∂P,dqdP)
            # push!(∂v∂P,dvdP)
            # push!(∂w∂P,dwdP)
        end

        #Derive the w expression
        # w_expr = zeros(Symbolics.Arr{Num, 1}, D, DX)
        # w_expr = Array{Num}(undef, D, DX)
        # for d in 1:D
        #     for dx in 1:DX
        #         w_expr[d,dx] = Symbolics.derivative(u_expr[d],x[dx])
        #     end
        # end

        # # Compute the derivatives of w with respect to P
        # ∂w_expr∂P = Array{Vector{Num}}(undef, D, DX)
        # ∂w∂P = Array{Vector{Function}}(undef, D, DX)
        # for d in 1:D
        #     for dx in 1:DX
        #         ∂w_expr∂P[d, dx] = [Symbolics.derivative(w_expr[d, dx], P[j]) for j in 1:NP]
        #         ∂w∂P[d, dx] = Symbolics.eval(build_function(∂w_expr∂P[d, dx], P, t, x))
        #     end
        # end


        # Build the callable functions
        u = [eval(build_function(u_expr[d], P, t, x)) for d in 1:D]
        v = [eval(build_function(v_expr[d], P, t, x)) for d in 1:D]
        w = [eval(build_function(w_expr[d], P, t, x)) for d in 1:D]

        # w = zeros(Function, D,DX)
        # for d in 1:D
        #     for dx in 1:DX
        #         w[d,dx] = eval(build_function(w_expr[d,dx], P, t, x))
        #     end
        # end

        new(u_expr, v_expr, w_expr, P, ∂u∂P, ∂v∂P, ∂w∂P, u, v, w, NP, P_sizes)
    end
end