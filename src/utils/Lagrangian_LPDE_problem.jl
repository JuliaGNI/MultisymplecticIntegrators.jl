struct LPDE{invType <: OptionalInvariants,
    parType <: OptionalParameters,
    perType <: OptionalPeriodicity} <: GeometricEquation{invType,parType,perType} 
end

function Base.show(io::IO, ::LPDE)
    print(io, "Lagrangian PDE")
end

# const LPDEProblem = EquationProblem{LPDE}

struct LPDEProblem{superType<:GeometricEquation,dType<:Number,tType<:Real,LSType,ICSType,BCSType,EUType,EVType,EWType,ICST<:NamedTuple,PT<:NamedTuple,IT<:Union{Nothing,NamedTuple},LSAT<:Union{Function,Nothing}} <: GeometricProblem{superType, dType, tType}
    lagrangian_system::LSType
    D::Int
    ics_function::ICSType
    bcs_function::BCSType
    
    ics::ICST

    timespan::Tuple{Float64, Float64}
    timestep::Float64

    xspan::Tuple{Float64, Float64}
    xstep::Float64

    params::PT
    internal::IT

    exact_u::EUType
    exact_v::EVType
    exact_w::EWType
    least_squares_assemble::LSAT
    function LPDEProblem(lag_sys,ics_function,bcs_function,ics_values,tspan, tstep, xspan, xstep, params,exact_u,exact_v,exact_w,least_squares_assemble = nothing;internal = nothing)
        superType = LPDE
        tType = eltype(tspan)
        dType = eltype(ics_values.u)
        # ics_values = merge((t = tspan[begin],), ics_values)
        new{superType,dType,tType,typeof(lag_sys),typeof(ics_function),typeof(bcs_function),typeof(exact_u),typeof(exact_v),typeof(exact_w),typeof(ics_values),typeof(params),typeof(internal),typeof(least_squares_assemble)}(#
            lag_sys,
            lag_sys.D,
            ics_function,
            bcs_function,
            ics_values,
            tspan,
            tstep,
            xspan,
            xstep,
            params,
            internal,
            exact_u,
            exact_v,
            exact_w,
            least_squares_assemble
        )
    end
end

function Base.show(io::IO, problem::LPDEProblem)
    print(io, "\n Lagrangian PDE Problem with:\n")
    print(io, "   Dimension D: $(problem.D) \n")
    print(io, "   Time span: $(problem.timespan), timestep: $(problem.timestep) \n")
    print(io, "   Space span: $(problem.xspan), xstep: $(problem.xstep) \n")
    print(io, "   Parameters: $(keys(problem.params)) \n")
    print(io, "   Initial condition fields: $(keys(problem.ics)) \n")
    print(io, "   Has internal variables: $(problem.internal !== nothing) \n")
    print(io, "   Has least squares assemble: $(problem.least_squares_assemble !== nothing) \n")
end

datatype(problem::LPDEProblem) = eltype(problem.ics.u)
timetype(problem::LPDEProblem) = typeof(problem.timestep)
timestep(problem::LPDEProblem) = problem.timestep
spacestep(problem::LPDEProblem) = problem.xstep
timespan(problem::LPDEProblem) = problem.timespan
periodicity(problem::LPDEProblem) = (u = NullPeriodicity(), v = NullPeriodicity(), w = NullPeriodicity())
initial_conditions(problem::LPDEProblem) = merge((t = problem.timespan[begin],), problem.ics)
parameters(problem::LPDEProblem) = problem.params
compute_vectorfields!(vecfield, sol, prob::LPDEProblem) = nothing
_extrapolate!(newsol, oldsol, problem::LPDEProblem, extrap) = nothing
