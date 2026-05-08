struct LPDE{invType <: OptionalInvariants,
    parType <: OptionalParameters,
    perType <: OptionalPeriodicity} <: GeometricEquation{invType,parType,perType} 
end

# const LPDEProblem = EquationProblem{LPDE}

struct LPDEProblem{superType<:GeometricEquation,dType<:Number,tType<:Real,LSType,ICSType,BCSType,EUType,EVType,EWType} <: GeometricProblem{superType, dType, tType}
    lagrangian_system::LSType
    D::Int
    ics_function::ICSType
    bcs_function::BCSType
    
    ics::NamedTuple

    timespan::Tuple{Float64, Float64}
    timestep::Float64

    xspan::Tuple{Float64, Float64}
    xstep::Float64

    params::NamedTuple
    internal::Union{Nothing,NamedTuple}

    exact_u::EUType
    exact_v::EVType
    exact_w::EWType
    least_squares_assemble::Union{Function,Nothing}
    function LPDEProblem(lag_sys,ics_function,bcs_function,ics_values,tspan, tstep, xspan, xstep, params,exact_u,exact_v,exact_w,least_squares_assemble = nothing;internal = nothing)
        superType = LPDE
        tType = eltype(tspan)
        dType = eltype(ics_values.u)
        # ics_values = merge((t = tspan[begin],), ics_values)
        new{superType,dType,tType,typeof(lag_sys),typeof(ics_function),typeof(bcs_function),typeof(exact_u),typeof(exact_v),typeof(exact_w)}(#
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