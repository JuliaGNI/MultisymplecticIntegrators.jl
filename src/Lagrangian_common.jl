
struct LagrangianPDESystem
    L
    t 
    x
    u # alias for q
    v # alias for ut
    w # alias for ux
    parameters
    equations
    # functions
    function LagrangianPDESystem(L,t,x,u,v,w,params = NamedTuple();simplify = true, scalarize = true)

        Ls = scalarize ? Symbolics.scalarize(L) : L
        Ls = simplify ? Symbolics.simplify(Ls) : Ls

        Dt, Dx, Du, Dv, Dw = lagrangianPDE_derivatives(t,x,u,v,w)

        ∂L∂u = [expand_derivatives(du(Ls)) for du in Du]
        ∂L∂v = [expand_derivatives(dv(Ls)) for dv in Dv]
        ∂L∂w = [expand_derivatives(dw(Ls)) for dw in Dw]
        # ∂L∂u = expand_derivatives(Du(Ls))
        # ∂L∂v = expand_derivatives(Dv(Ls))
        # ∂L∂w = expand_derivatives(Dw(Ls))

        ∂_∂t_∂L∂v  = [expand_derivatives(Dt(dv(Ls))) for dv in Dv]
        ∂_∂x_∂L∂w = sum(hcat([expand_derivatives(dx.(∂L∂w)) for dx in Dx]...),dims=2)[:,1]
        EL_field = [∂L∂u[i] - ∂_∂t_∂L∂v[i] - ∂_∂x_∂L∂w[i] for i in eachindex(∂L∂u,∂_∂t_∂L∂v,∂_∂x_∂L∂w)]

        # ∂_∂t_∂L∂v = expand_derivatives(Dt(Dv(Ls)))
        # ∂_∂x_∂L∂w = expand_derivatives(Dx(Dw(Ls)))
        # EL_field = ∂L∂u - ∂_∂t_∂L∂v - ∂_∂x_∂L∂w
        equs = (
            L = Ls,
            ∂L∂u = ∂L∂u,
            ∂L∂v = ∂L∂v,
            ∂L∂w = ∂L∂w,
            ∂_∂t_∂L∂v = ∂_∂t_∂L∂v,
            ∂_∂x_∂L∂w = ∂_∂x_∂L∂w,
            EL = EL_field,
        )
        return new(Ls, t, x, u, v, w, params, equs)
    end
end



function lagrangianPDE_variables(variable_dimension::Integer,x_domain_dimension::Integer)
    @variables t
    @variables x[1:x_domain_dimension]
    @variables (u(x...,t))[1:variable_dimension]     #@variables (u(sym_x,sym_t))[1:variable_dimension] to not expand spatial variable x
    @variables (v(x...,t))[1:variable_dimension]
    @variables (w(x...,t))[1:variable_dimension,1:x_domain_dimension] # what is the dimension of w?

    return (t, x, u, v, w)
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
