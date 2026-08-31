struct SindyPDEBasis{UPT, VPT, WPT, UFT, VFT, WFT} <: AbstractPDEBasis
    u_expr::Vector{Num} # expression
    v_expr::Vector{Num}
    w_expr::Matrix{Num}

    expr_params::Vector{Symbolics.Arr{Num, 1}} # parameters from the expression

    ∂u∂P::UPT # derivatives with respect to P
    ∂v∂P::VPT
    ∂w∂P::WPT

    u::UFT # callable function
    v::VFT
    w::WFT

    NP::Int
    P_sizes::Vector{Int}

    function SindyPDEBasis(u_expr::Vector{Num}, P::Vector{Symbolics.Arr{Num, 1}}, t::Num, x::Vector{Num})
        P_sizes = map(length, P)
        NP = sum(P_sizes)
        D = length(u_expr)
        DX = length(x)

        v_expr = [Symbolics.derivative(u_expr[i], t) for i in 1:D]
        w_expr = [Symbolics.derivative(u_expr[i], x[j]) for i in 1:D, j in 1:DX]

        # Compute the derivatives of u and v with respect to P
        ∂u∂P = []
        ∂v∂P = []
        ∂w∂P = []
        for d in 1:D
            ∂u_expr∂P = zeros(Num, P_sizes[d])
            ∂v_expr∂P = zeros(Num, P_sizes[d])
            ∂w_expr∂P = zeros(Num, P_sizes[d])
            for i in 1:P_sizes[d]
                ∂u_expr∂P[i] = Symbolics.derivative(u_expr[d], P[d][i])
                ∂v_expr∂P[i] = Symbolics.derivative(v_expr[d], P[d][i])
                ∂w_expr∂P[i] = Symbolics.derivative(w_expr[d], P[d][i])
            end
            push!(∂u∂P, [eval(build_function(∂u_expr∂P[i], P, t, x)) for i in 1:P_sizes[d]])
            push!(∂v∂P, [eval(build_function(∂v_expr∂P[i], P, t, x)) for i in 1:P_sizes[d]])
            push!(∂w∂P, [eval(build_function(∂w_expr∂P[i], P, t, x)) for i in 1:P_sizes[d]])
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

        new{typeof(∂u∂P), typeof(∂v∂P), typeof(∂w∂P), typeof(u), typeof(v), typeof(w)}(
            u_expr, v_expr, w_expr, P, ∂u∂P, ∂v∂P, ∂w∂P, u, v, w, NP, P_sizes)
    end
end

function Base.show(io::IO, basis::SindyPDEBasis)
    print(io, "\n SINDy PDE Basis with:\n")
    print(io, "   Equation dimension D: $(length(basis.u_expr)) \n")
    print(io, "   Number of optimized parameters NP: $(basis.NP) \n")
    print(io, "   Parameter block sizes: $(basis.P_sizes) \n")
end
