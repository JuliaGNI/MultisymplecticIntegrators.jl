
struct LPDESystem
    L
    t 
    x
    u # alias for q
    v # alias for ut
    w # alias for ux
    parameters
    equations
    functions
    function LPDESystem(L,t,x,U,V,W,params = NamedTuple();simplify = true, scalarize = true)

        DX = length(x)
        D = length(U)

        Ls = scalarize ? Symbolics.scalarize(L) : L
        Ls = simplify ? Symbolics.simplify(Ls) : Ls

        ∂L∂U_expr = [Symbolics.derivative(Ls, U[i]) for i in eachindex(U)]
        ∂L∂V_expr = [Symbolics.derivative(Ls, V[i]) for i in eachindex(V)]
        
        ∂L∂W_expr = zeros(Num,D,DX)
        for d in 1:D
            for dx in 1:DX
                ∂L∂W_expr[d,dx] = Symbolics.derivative(Ls, W[d,dx])
            end
        end

        equs = (
            L = Ls,
            ∂L∂U = ∂L∂U_expr,
            ∂L∂V = ∂L∂V_expr,
            ∂L∂W = ∂L∂W_expr,
        ) # set of expressions

        ∂L∂U = [Symbolics.eval(substitute_parameters(Symbolics.build_function(∂L∂U_expr[i], U, V, W,sparams...; nanmath = false),sparams)) for i in eachindex(∂L∂U_expr)]
        ∂L∂V = [Symbolics.eval(substitute_parameters(Symbolics.build_function(∂L∂V_expr[i], U, V, W,sparams...; nanmath = false),sparams)) for i in eachindex(∂L∂V_expr)]

        ∂L∂W = zeros(D,DX)
        for d in 1:D
            for dx in 1:DX
                ∂L∂W[d,dx] = Symbolics.eval(substitute_parameters(Symbolics.build_function(∂L∂W_expr[d,dx], U, V, W,sparams...; nanmath = false),sparams))
            end
        end

        codes = (
            ∂L∂U = ∂L∂U,
            ∂L∂V = ∂L∂V,
            ∂L∂W = ∂L∂W,
        ) # set of callable functions

        return new(Ls, t, x, U,V,W, params, equs, codes)
    end
end



function LPDE_variables(variable_dimension::Integer,x_domain_dimension::Integer)
    @variables t
    @variables x[1:x_domain_dimension]
    # @variables (u(x...,t))[1:variable_dimension]     #@variables (u(sym_x,sym_t))[1:variable_dimension] to not expand spatial variable x
    # @variables (v(x...,t))[1:variable_dimension]
    # @variables (w(x...,t))[1:variable_dimension,1:x_domain_dimension] # what is the dimension of w?
    @variables U[1:variable_dimension]     
    @variables V[1:variable_dimension]
    @variables W[1:variable_dimension,1:x_domain_dimension]
    return (t, x, U, V, W)
end

function lagrangianPDE_derivatives(t,x,u,v,w)

    Dt = Differential(t)
    Dx = collect(Differential.(x))
    Du = collect(Differential.(u))
    Dv = collect(Differential.(v))
    Dw = collect(Differential.(w))
    # Dx = Differential(x)
    # Du = Differential(u)
    # Dv = Differential(v)
    # Dw = Differential(w)

    return (Dt, Dx, Du, Dv, Dw)
end

