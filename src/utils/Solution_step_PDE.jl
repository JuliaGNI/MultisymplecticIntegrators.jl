struct SolutionStepPDE{TT,UT,VT,WT,XT}
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
    function SolutionStepPDE(t::TT,u::UT,v::VT,w::WT,x,::XT,params::PT; nhistory = 2,internal::IT = NamedTuple()) where {TT,UT,VT,WT,XT,PT,IT}
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