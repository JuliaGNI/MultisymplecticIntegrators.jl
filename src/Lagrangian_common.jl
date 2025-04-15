
struct LagrangianPDESystem
    L
    t 
    x
    u # alias for q
    v # alias for ut
    w # alias for ux
    parameters
    equations
    functions
    function LagrangianPDESystem(L,t,x,U,V,W,params = NamedTuple();simplify = true, scalarize = true)

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



function lagrangianPDE_variables(variable_dimension::Integer,x_domain_dimension::Integer)
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


function symbolize(p::Union{AbstractArray, Tuple}, name)
    vars = @variables $(name)[axes(p)...]
    first(vars)
end

parameter(name::Symbol) = Num(Symbolics.Sym{Real}(name))

function symbolize(::Number, name)
    parameter(name)
end

function symbolize(p::Union{Symbolics.Num, Symbolics.Arr{Symbolics.Num}}, name)
    p
end

function symbolize(params::NamedTuple)
    NamedTuple{keys(params)}(Tuple(symbolize(v, Symbol("$(k)ₚ")) for (k,v) in pairs(params)))
end


function substitute_parameters(code, params)
    if length(params) > 0
        # generate string of parameter arguments as they appear in generated code
        paramstr = string(Tuple(string(k) * "ₚ, " for k in keys(params))...)
        # remove trailing comma
        paramstr = paramstr[begin:prevind(paramstr, first((findlast(",", paramstr))))]
        # convert code expression to string
        code_str = string(code)
        # extract function header
        func_str = code_str[first(findfirst("function (", code_str)):last(findfirst(paramstr * ")", code_str))]
        # replace parameter list in function header with `params`
        func_str_params = replace(func_str, paramstr * ")" => "params)")
        # replace function header in code
        code_str = replace(code_str, func_str => func_str_params)
        # replace all params with named tuple entries
        for k in keys(params)
            code_str = replace(code_str, "$(k)ₚ" => "params.$(k)")
        end
        # convert code string back to expression
        return Meta.parse(code_str)
    else
        # convert code expression to string
        code_str = string(code)
        # extract function header
        func_str = code_str[first(findfirst("function (", code_str)):last(findfirst(")", code_str))]
        # append params argument to function header
        func_str_params = replace(func_str, ")" => ", params)")
        # replace function header in code
        code_str = replace(code_str, func_str => func_str_params)
        # convert code string back to expression
        return Meta.parse(code_str)
    end
end


struct SolutionStepLPDE{TT,UT,VT,WT,XT}
    t::TT
    history

    u::UT
    v::VT
    w::WT   
    x::XT

    ū::UT
    v̄::VT
    w̄::WT
    x̄::XT

    params::NamedTuple
    internal
    function SolutionStepLPDE(t::TT,u::UT,v::VT,w::WT,x,::XT,params::PT; nhistory = 2,internal::IT = NamedTuple()) where {TT,UT,VT,WT,XT,PT,IT}
        @assert nhistory ≥ 1 "nhistory must be greater than 0"

        history = (
            t = OffsetVector([zero(t) for _ in 0:nhistory], 0:nhistory),
            u = OffsetVector([zero(u) for _ in 0:nhistory], 0:nhistory),
            v = OffsetVector([zero(v) for _ in 0:nhistory], 0:nhistory),
            w = OffsetVector([zero(w) for _ in 0:nhistory], 0:nhistory),
            x = OffsetVector([zero(x) for _ in 0:nhistory], 0:nhistory),
        )

        u = history.u[0]
        v = history.v[0]
        w = history.w[0]
        x = history.x[0]

        ū = history.u[1]
        v̄ = history.v[1]
        w̄ = history.w[1]
        x̄ = history.x[1]

        return new{typeof(t),typeof(u),typeof(v),typeof(w),typeof(x)}(t, history, u, v, w, x, ū, v̄, w̄, x̄, params, internal)
    end
end