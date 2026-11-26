mutable struct _LindbladWorkspace
    left::Matrix{ComplexF64}
end


_LindbladWorkspace(dim::Integer) = _LindbladWorkspace(zeros(ComplexF64, dim, dim))

struct _LindbladAction{H,C,D}
    hamiltonian::H
    jumps::C
    products::D
    workspace::_LindbladWorkspace
    dimension::Int
end


Base.size(L::_LindbladAction) = (L.dimension^2, L.dimension^2)
Base.size(L::_LindbladAction, k::Integer) = k > 0 ? (k <= 2 ? L.dimension^2 : 1) : throw(ArgumentError("dimension must be positive"))
Base.eltype(::_LindbladAction) = ComplexF64
Base.eltype(::Type{<:_LindbladAction}) = ComplexF64
Base.copy(L::_LindbladAction) = _LindbladAction(L.hamiltonian, L.jumps, L.products, _LindbladWorkspace(L.dimension), L.dimension)

function LinearAlgebra.mul!(y::AbstractVector{<:Number}, L::_LindbladAction, x::AbstractVector{<:Number})
    return mul!(y, L, x, 1, 0)
end


function LinearAlgebra.mul!(y::AbstractVector{<:Number}, L::_LindbladAction, x::AbstractVector{<:Number}, alpha::Number, beta::Number)
    dim = L.dimension
    if iszero(alpha)
        return iszero(beta) ? fill!(y, zero(eltype(y))) : rmul!(y, beta)
    end
    
    rho = reshape(Base.mightalias(x, y) ? copy(x) : x, dim, dim)
    derivative = reshape(y, dim, dim)
    left = L.workspace.left
    mul!(derivative, L.hamiltonian, rho, -1im * alpha, beta)
    mul!(derivative, rho, L.hamiltonian, 1im * alpha, 1)
    for (C, D) in zip(L.jumps, L.products)
        mul!(left, C, rho)
        mul!(derivative, left, C', alpha, 1)
        mul!(derivative, D, rho, -0.5 * alpha, 1)
        mul!(derivative, rho, D, -0.5 * alpha, 1)
    end
    
    return y
end


Base.:*(L::_LindbladAction, x::AbstractVector{<:Number}) = mul!(zeros(ComplexF64, size(L, 1)), L, x)

function _explicit_liouvillian(H::AbstractMatrix{<:Number}, jumps, products, representation::Symbol)
    representation in (:dense, :sparse) || throw(ArgumentError("representation must be :dense or :sparse"))
    convert_matrix = representation == :sparse ? A -> sparse(ComplexF64.(A)) : A -> Matrix{ComplexF64}(A)
    dim = size(H, 1)
    eye = representation == :sparse ? spdiagm(0 => ones(ComplexF64, dim)) : Matrix{ComplexF64}(I, dim, dim)
    hamiltonian = convert_matrix(H)
    generator = -1im * (kron(eye, hamiltonian) - kron(transpose(hamiltonian), eye))
    for (C0, D0) in zip(jumps, products)
        C, D = convert_matrix(C0), convert_matrix(D0)
        generator += kron(conj(C), C) - 0.5 * (kron(eye, D) + kron(transpose(D), eye))
    end
    
    return representation == :sparse ? dropzeros!(generator) : generator
end


Base.Matrix(L::_LindbladAction) = _explicit_liouvillian(L.hamiltonian, L.jumps, L.products, :dense)
SparseArrays.sparse(L::_LindbladAction) = _explicit_liouvillian(L.hamiltonian, L.jumps, L.products, :sparse)

"""
    liouvillian(H, jump_operators=(); representation=:sparse)

Construct the generator of `d vec(rho) / dt = L * vec(rho)`. The dissipator is
`C * rho * C' - (C' * C * rho + rho * C' * C) / 2` for each operator `C`.

`:sparse` and `:dense` return explicit matrices. `:matrixfree` returns an operator
supporting `size`, `eltype`, `mul!`, and multiplication by a vector.

`Matrix(L)` or `sparse(L)` to conver to matrices.
"""
function liouvillian(H::AbstractMatrix{<:Number}, jump_operators = (); representation::Symbol = :sparse)
    dim = size(H, 1)
    dim > 0 || throw(ArgumentError("Hamiltonian must be nonempty"))
    hamiltonian = _checked_hamiltonian(H, dim)
    jumps = _checked_jumps(jump_operators, dim)
    products = [C' * C for C in jumps]
    if representation == :matrixfree
        return _LindbladAction(hamiltonian, jumps, products, _LindbladWorkspace(dim), dim)
    end
    
    return _explicit_liouvillian(hamiltonian, jumps, products, representation)
end


struct _LindbladModel{H,C,D,G,S,B} <: _EvolutionModel
    hamiltonian::H
    jumps::C
    products::D
    generator::G
    dissipator::S
    buffer::B
    dimension::Int
    representation::Symbol
    workspace::_LindbladWorkspace
end


function _lindblad_generator(H::AbstractMatrix{<:Number}, jumps, products, representation::Symbol, workspace::_LindbladWorkspace)
    return representation == :matrixfree ? _LindbladAction(H, jumps, products, workspace, size(H, 1)) : _explicit_liouvillian(H, jumps, products, representation)
end


function _lindblad_model(H, jump_operators, dim::Integer, representation::Symbol)
    representation in (:matrixfree, :sparse, :dense) || throw(ArgumentError("representation must be :matrixfree, :sparse, or :dense"))
    H0 = _checked_hamiltonian(H isa AbstractMatrix ? H : H(0.0), dim)
    dynamic_jumps = applicable(jump_operators, 0.0)
    jumps = dynamic_jumps ? jump_operators : _checked_jumps(jump_operators, dim)
    initial_jumps = dynamic_jumps ? _checked_jumps(jumps(0.0), dim) : jumps
    products = dynamic_jumps ? nothing : [C' * C for C in jumps]
    workspace = _LindbladWorkspace(dim)
    generator = H isa AbstractMatrix && !dynamic_jumps ? _lindblad_generator(H0, initial_jumps, products, representation, workspace) : nothing
    dissipator = !(H isa AbstractMatrix) && !dynamic_jumps && !isempty(initial_jumps) && representation != :matrixfree ?
                 _explicit_liouvillian(spzeros(ComplexF64, dim, dim), initial_jumps, products, representation) : nothing
    
    buffer = !(H isa AbstractMatrix) && !dynamic_jumps && representation == :dense ? Matrix{ComplexF64}(undef, dim^2, dim^2) : nothing
    
    return _LindbladModel(H, jumps, products, generator, dissipator, buffer, dim^2, representation, workspace)
end


function _commutator_generator!(destination::Matrix{ComplexF64}, H::AbstractMatrix{<:Number}, dissipator)
    dim = size(H, 1)
    size(H, 2) == dim || throw(DimensionMismatch("Hamiltonian must be square"))
    size(destination) == (dim^2, dim^2) || throw(DimensionMismatch("Liouvillian buffer must match the Hamiltonian"))
    if dissipator === nothing
        fill!(destination, 0)
    else
        copyto!(destination, dissipator)
    end

    @inbounds for j in 1:dim, k in 1:dim
        coefficient = 1im * H[k, j]
        for i in 1:dim
            destination[i + (j - 1) * dim, i + (k - 1) * dim] += coefficient
            destination[k + (i - 1) * dim, j + (i - 1) * dim] -= coefficient
        end
    end
    
    return destination
end


function _generator(model::_LindbladModel, t::Real)
    model.generator !== nothing && return model.generator
    dim = size(model.workspace.left, 1)
    H = _checked_hamiltonian(model.hamiltonian isa AbstractMatrix ? model.hamiltonian : model.hamiltonian(t), dim)
    model.buffer !== nothing && return _commutator_generator!(model.buffer, H, model.dissipator)
    if model.dissipator !== nothing
        generator = _explicit_liouvillian(H, (), (), model.representation)
        generator .+= model.dissipator
        
        return generator
    end
    
    dynamic_jumps = model.products === nothing
    jumps = dynamic_jumps ? _checked_jumps(model.jumps(t), dim) : model.jumps
    products = dynamic_jumps ? [C' * C for C in jumps] : model.products
    
    return _lindblad_generator(H, jumps, products, model.representation, model.workspace)
end


function _check_density(rho::AbstractMatrix{<:Number}, tolerance::Real, exception_type::Type{<:Exception})
    all(isfinite, rho) || throw(exception_type("density matrix must be finite"))
    isapprox(rho, rho'; atol = tolerance, rtol = tolerance) || throw(exception_type("density matrix must be Hermitian"))
    isapprox(tr(rho), 1; atol = tolerance, rtol = tolerance) || throw(exception_type("density matrix must have unit trace"))
    eigmin(Hermitian(rho)) >= -tolerance || throw(exception_type("density matrix must be positive semidefinite"))
    
    return nothing
end


function _density_state(basis::Basis, initial::AbstractMatrix{<:Number})
    dim = length(basis)
    size(initial) == (dim, dim) || throw(DimensionMismatch("density matrix must match basis"))
    rho = Matrix{ComplexF64}(initial)
    _check_density(rho, 1e-10, ArgumentError)
    
    return vec(rho)
end

"""
    run_lindblad(basis, H, initial_density, outputs, end_time;
                 jump_operators = (), representation = :matrixfree, saveat = nothing,
                 saves = 1000, dt = nothing, steps = 0, method = :auto, krylov_dimension = 0,
                 atol = 1e-10, rtol = 1e-8, max_steps = 10^7, store_states = false,
                 check_physical = true)

Deterministic Lindblad evolution of a density matrix.

Output accepts functions of `(rho, t)` or `(rho)`. 

Return `(times, outputs, final_state, states)`.
"""
function run_lindblad(basis::Basis, H, initial_density::AbstractMatrix{<:Number}, outputs, end_time::Real;
                      jump_operators = (), representation::Symbol = :matrixfree, saveat = nothing, saves::Integer = 1000,
                      dt::Union{Nothing,Real} = nothing, steps::Integer = 0, method::Symbol = :auto,
                      krylov_dimension::Integer = 0, atol::Real = 1e-10, rtol::Real = 1e-8,
                      max_steps::Integer = 10^7, store_states::Bool = false, check_physical::Bool = true)
        
    times = _save_times(end_time, saveat, saves)
    initial = _density_state(basis, initial_density)
    options = _solver_options(end_time, times; dt, steps, method, krylov_dimension, atol, rtol, max_steps)
    dim = length(basis)
    state_view = state -> reshape(state, dim, dim)
    callbacks = _output_callbacks(outputs, state_view(initial))
    model = _lindblad_model(H, jump_operators, dim, representation)
    result = _deterministic_evolution(model, initial, callbacks, times, options; state_view, store_states)
    check_physical && _check_density(result.final_state, max(1e-10, 10 * (atol + rtol)), ErrorException)
    
    return result
end
