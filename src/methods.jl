using GeometricIntegratorsBase:GeometricMethod
abstract type PDEMethod <: GeometricMethod end

# using GeometricBase:AbstractProblem
# abstract type PDEProblem <: AbstractProblem end


using GeometricEquations: GeometricEquation
using GeometricBase:OptionalInvariants,OptionalParameters,OptionalPeriodicity


using GeometricIntegratorsBase:IntegratorCache
abstract type PDEIntegratorCache{DT,D} <: IntegratorCache{DT,D} end

using GeometricBase:AbstractIntegrator
abstract type AbstractPDEIntegrator <: AbstractIntegrator end

# using GeometricBase:AbstractSolution
# abstract type AbstractPDESolution <: AbstractSolution end

# using CompactBasisFunctions: Basis
abstract type AbstractPDEBasis end #<: Basis 

abstract type InitialParametersMethod end
struct LSGD <: InitialParametersMethod end
# struct OGA2D <: InitialParametersMethod end
struct ELM <: InitialParametersMethod end
struct PINN <: InitialParametersMethod end
struct TrialOGA2D <: InitialParametersMethod end

_show_typename(io::IO, obj) = print(io, nameof(typeof(obj)))
_hasfield(obj, field::Symbol) = field in fieldnames(typeof(obj))

function _show_field(io::IO, obj, field::Symbol; label::AbstractString=String(field))
    _hasfield(obj, field) || return false
    print(io, "   ", label, ": ", getfield(obj, field), "\n")
    return true
end

function _show_type_field(io::IO, obj, field::Symbol; label::AbstractString=String(field))
    _hasfield(obj, field) || return false
    print(io, "   ", label, ": ", nameof(typeof(getfield(obj, field))), "\n")
    return true
end

_points_per_interval(npoints::Integer, nintervals::Integer) = nintervals > 0 && npoints % nintervals == 0 ? npoints ÷ nintervals : missing

function Base.show(io::IO, method::InitialParametersMethod)
    _show_typename(io, method)
end

function Base.show(io::IO, basis::AbstractPDEBasis)
    print(io, "\n ")
    _show_typename(io, basis)
    print(io, " with:\n")
    _show_field(io, basis, :S; label="basis functions")
    _show_field(io, basis, :NP; label="parameters")
    _show_field(io, basis, :optim_mode)
    _show_field(io, basis, :activation_function)
end

function Base.show(io::IO, method::PDEMethod)
    print(io, "\n ")
    _show_typename(io, method)
    print(io, " with:\n")
    _show_type_field(io, method, :basis)
    _show_type_field(io, method, :symbolic_expr_basis; label="basis")
    _show_field(io, method, :RT; label="time quadrature points")
    _show_field(io, method, :RX; label="space quadrature points")
    _show_field(io, method, :t_num_interval; label="time intervals")
    _show_field(io, method, :x_num_interval; label="space intervals")
    _show_field(io, method, :Nbasis_μ_t)
    _show_field(io, method, :Nbasis_λ_x)
    _show_type_field(io, method, :initial_guess_method)
    _show_field(io, method, :show_status)
end
