struct LPDE{invType <: OptionalInvariants,
    parType <: OptionalParameters,
    perType <: OptionalPeriodicity} <: GeometricEquation{invType,parType,perType} end

# const LPDEProblem = EquationProblem{LPDE} ? 

struct LPDEProblem{superType<:GeometricEquation,} <: GeometricProblem{superType}
    lagrangian_system
    D::Int
    ics_function::Function
    bcs_function::Function
    
    ics::NamedTuple

    timespan::Tuple
    timestep

    xspan::Tuple{Float64, Float64}
    xstep::Float64

    params
    internal

    exact_u
    exact_v
    exact_w
    least_squares_assemble
    function LPDEProblem(lag_sys,ics_function,bcs_function,ics_values,tspan, tstep, xspan, xstep, params,exact_u,exact_v,exact_w,least_squares_assemble = nothing;internal = nothing)
        superType = LPDE
        # ics_values = merge((t = tspan[begin],), ics_values)
        new{superType, }(
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