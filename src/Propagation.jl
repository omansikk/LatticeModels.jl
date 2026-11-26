using LinearAlgebra, SparseArrays


abstract type _EvolutionModel end
abstract type _Propagation end

struct _StepBuffers
    full::Vector{ComplexF64}
    refined::Vector{ComplexF64}
    halfway::Vector{ComplexF64}
    trial::Vector{ComplexF64}
    event::Vector{ComplexF64}
    observed::Vector{ComplexF64}
end

_StepBuffers(dim::Integer) = _StepBuffers((zeros(ComplexF64, dim) for i in 1:6)...)

mutable struct _DensePropagation <: _Propagation
    exponentials::Dict{Float64,Matrix{ComplexF64}}
    scaled::Matrix{ComplexF64}
    steps::_StepBuffers
end


mutable struct _ArnoldiPropagation <: _Propagation
    vectors::Matrix{ComplexF64}
    projection::Matrix{ComplexF64}
    residual::Vector{ComplexF64}
    scaled::Matrix{ComplexF64}
    steps::_StepBuffers
end


function _checked_hamiltonian(H::AbstractMatrix{<:Number}, dim::Integer)
    size(H) == (dim, dim) || throw(DimensionMismatch("Hamiltonian must match the state dimension"))
    all(isfinite, H) || throw(ArgumentError("Hamiltonian must contain finite entries"))
    isapprox(H, H'; atol = 1e-12, rtol = 1e-12) || throw(ArgumentError("Hamiltonian must be Hermitian; pass jump operators separately"))
    
    return H
end


function _checked_jumps(jumps, dim::Integer)
    operators = collect(jumps)
    for C in operators
        C isa AbstractMatrix{<:Number} || throw(ArgumentError("jump operators must be numeric matrices"))
        size(C) == (dim, dim) || throw(DimensionMismatch("jump operators must match the state dimension"))
        all(isfinite, C) || throw(ArgumentError("jump operators must contain finite entries"))
    end
    
    return operators
end


function _propagation_workspace(model::_EvolutionModel, method::Symbol, krylov_dimension::Integer)
    method in (:auto, :dense, :krylov) || throw(ArgumentError("method must be :auto, :dense, or :krylov"))
    generator = _generator(model, 0.0)
    method == :dense && !(generator isa AbstractMatrix) && throw(ArgumentError("dense propagation requires an explicit generator; select representation = :dense or :sparse"))
    use_krylov = method == :krylov || (method == :auto && (!(generator isa AbstractMatrix) || issparse(generator)))
    dim = model.dimension
    steps = _StepBuffers(dim)
    if use_krylov
        m = krylov_dimension == 0 ? min(30, dim) : min(krylov_dimension, dim)
        m >= min(2, dim) || throw(ArgumentError("Krylov dimension must be at least min(2, state dimension)"))
        return _ArnoldiPropagation(zeros(ComplexF64, dim, m + 1), zeros(ComplexF64, m + 1, m), zeros(ComplexF64, dim), zeros(ComplexF64, m, m), steps)
    end
    
    return _DensePropagation(Dict{Float64,Matrix{ComplexF64}}(), zeros(ComplexF64, dim, dim), steps)
end


function _exponential_action!(destination::Vector{ComplexF64}, generator::AbstractMatrix{<:Number},
    state::Vector{ComplexF64}, h::Float64, work::_DensePropagation, cache::Bool)
    if !cache
        work.scaled .= h .* generator
        exponential = LinearAlgebra.exp!(work.scaled)
        return mul!(destination, exponential, state)
    end
    
    cache_limit = max(1, min(8, (64 * 1024^2) ÷ (sizeof(ComplexF64) * length(generator))))
    length(work.exponentials) >= cache_limit && !haskey(work.exponentials, h) && empty!(work.exponentials)
    exponential = get!(work.exponentials, h) do
        scaled = Matrix{ComplexF64}(undef, size(generator))
        scaled .= h .* generator
        LinearAlgebra.exp!(scaled)
    end
    
    return mul!(destination, exponential, state)
end


function _exponential_action!(destination::Vector{ComplexF64}, generator, state::Vector{ComplexF64},
    h::Float64, work::_ArnoldiPropagation, cache::Bool)
    beta = norm(state)
    iszero(beta) && return copyto!(destination, state)
    V, P, residual = work.vectors, work.projection, work.residual
    fill!(P, 0)
    V[:, 1] .= state ./ beta
    m = size(P, 2)
    used = m
    for j in 1:m
        mul!(residual, generator, @view(V[:, j]))
        original_norm = norm(residual)
        for pass in 1:2, i in 1:j
            coefficient = dot(@view(V[:, i]), residual)
            P[i, j] += coefficient
            @views residual .-= coefficient .* V[:, i]
        end
        
        remainder = norm(residual)
        if remainder <= 32eps(Float64) * max(1.0, original_norm)
            used = j
            break
        end
        
        P[j + 1, j] = remainder
        V[:, j + 1] .= residual ./ remainder
    end
    
    scaled = @view(work.scaled[1:used, 1:used])
    scaled .= h .* @view(P[1:used, 1:used])
    exponential = LinearAlgebra.exp!(scaled)
    
    return mul!(destination, @view(V[:, 1:used]), @view(exponential[:, 1]), beta, 0)
end

function _exponential_action(generator, state::Vector{ComplexF64}, h::Float64, work::_Propagation, cache::Bool)
    return _exponential_action!(similar(state), generator, state, h, work, cache)
end


function _midpoint_step!(destination::Vector{ComplexF64}, model::_EvolutionModel, state::Vector{ComplexF64},
    t::Float64, h::Float64, work::_Propagation)
    
    return _exponential_action!(destination, _generator(model, t + h / 2), state, h, work, model.generator !== nothing)
end


function _two_half_steps!(destination::Vector{ComplexF64}, model::_EvolutionModel, state::Vector{ComplexF64},
    t::Float64, h::Float64, work::_Propagation)
    halfway = work.steps.halfway
    _midpoint_step!(halfway, model, state, t, h / 2, work)
    
    return _midpoint_step!(destination, model, halfway, t + h / 2, h / 2, work)
end


function _accepted_step(model::_EvolutionModel, state::Vector{ComplexF64}, t::Float64, h::Float64,
    work::_Propagation, atol::Real, rtol::Real; contractive::Bool = false)
    full, refined = work.steps.full, work.steps.refined
    if model.generator !== nothing && work isa _DensePropagation
        return _midpoint_step!(refined, model, state, t, h, work), h, 2h
    end
    
    for attempt in 1:60
        _midpoint_step!(full, model, state, t, h, work)
        _two_half_steps!(refined, model, state, t, h, work)
        tolerance = atol + rtol * max(norm(state), norm(refined))
        full .-= refined
        error = norm(full)
        if error <= tolerance && (!contractive || sum(abs2, refined) <= sum(abs2, state) + tolerance)
            return refined, h, error < tolerance / 8 ? 2h : h
        end
        
        h /= 2
        h > 16eps(max(1.0, abs(t))) || throw(ErrorException("propagation failed to converge; increase krylov_dimension or relax tolerances"))
    end
    
    throw(ErrorException("propagation failed to converge"))
end


function _next_step_size(proposed::Float64, requested::Float64, accepted::Float64, suggested::Float64)
    return requested < proposed && accepted == requested ? max(proposed, suggested) : suggested
end


function _observed_state(state::Vector{ComplexF64}, state_view, work::_Propagation)
    state_view === normalize || return state_view(state)
    copyto!(work.steps.observed, state)
    return normalize!(work.steps.observed)
end


_generator_buffer(H::StridedMatrix{<:Number}) = Matrix{ComplexF64}(undef, size(H))
_generator_buffer(H::SparseMatrixCSC{<:Number}) = spzeros(ComplexF64, size(H)...)
_generator_buffer(H::AbstractMatrix{<:Number}) = nothing

function _reuse_generator!(buffer, H::AbstractMatrix{<:Number}, scale::Number, offset)
    return offset === nothing ? scale * H : scale * H + offset
end


function _reuse_generator!(buffer::Matrix{ComplexF64}, H::StridedMatrix{<:Number}, scale::Number, offset)
    if offset === nothing
        buffer .= scale .* H
    else
        buffer .= scale .* H .+ offset
    end
    
    return buffer
end


function _reuse_generator!(buffer::SparseMatrixCSC{ComplexF64,Int}, H::SparseMatrixCSC{<:Number}, scale::Number,
    offset::Union{Nothing,SparseMatrixCSC})
    if offset === nothing
        buffer .= scale .* H
    else
        buffer .= scale .* H .+ offset
    end
    
    return buffer
end


function _output_callbacks(outputs, initial::AbstractArray{<:Number})
    return map(collect(outputs)) do output
        if applicable(output, initial, 0.0)
            (psi, t) -> output(psi, t)
        elseif applicable(output, initial)
            (psi, t) -> output(psi)
        else
            throw(ArgumentError("outputs must accept (state, time) or (state)"))
        end
    end
end


function _initial_outputs(callbacks, state::AbstractArray{<:Number}, count::Integer)
    return map(callbacks) do callback
        value = callback(state, 0.0)
        value isa Number && isfinite(value) || throw(ArgumentError("outputs must return finite numeric scalars"))
        values = Vector{typeof(float(value))}(undef, count)
        values[1] = value
        values
    end
end


function _save_times(end_time::Real, saveat, saves::Integer)
    isfinite(end_time) && end_time >= 0 || throw(ArgumentError("end_time must be finite and nonnegative"))
    if saveat === nothing
        saves > 0 || throw(ArgumentError("saves must be positive"))
        return end_time == 0 ? [0.0] : collect(range(0.0, Float64(end_time); length = saves + 1))
    end
    
    times = Float64.(collect(saveat))
    all(t -> isfinite(t) && 0 <= t <= end_time, times) || throw(ArgumentError("saveat times must lie between zero and end_time"))
    issorted(times) && allunique(times) || throw(ArgumentError("saveat times must be strictly increasing"))
    
    return unique(vcat(0.0, times, Float64(end_time)))
end


function _solver_options(end_time::Real, times::Vector{Float64}; dt::Union{Nothing,Real}, steps::Integer, method::Symbol, 
                         krylov_dimension::Integer, atol::Real, rtol::Real, max_steps::Integer)
                         
    steps >= 0 && krylov_dimension >= 0 && max_steps > 0 || throw(ArgumentError("invalid step or Krylov limits"))
    method in (:auto, :dense, :krylov) || throw(ArgumentError("method must be :auto, :dense, or :krylov"))
    dt !== nothing && steps != 0 && throw(ArgumentError("specify dt or steps, not both"))
    step_size = dt === nothing ? (steps > 0 && end_time > 0 ? Float64(end_time) / steps : max(Float64(end_time), 1.0) / max(100, length(times) - 1)) : Float64(dt)
    isfinite(step_size) && step_size > 0 || throw(ArgumentError("dt must be finite and positive"))
    all(x -> isfinite(x) && x > 0, (atol, rtol)) || throw(ArgumentError("tolerances must be finite and positive"))
    
    return (; dt = step_size, method, krylov_dimension, atol, rtol, max_steps)
end


function _pure_state(basis::Basis, initial::AbstractVector{<:Number})
    length(initial) == length(basis) || throw(DimensionMismatch("state length must match basis"))
    state = Vector{ComplexF64}(initial)
    all(isfinite, state) && norm(state) > 0 || throw(ArgumentError("initial state must be finite and nonzero"))
    
    return normalize!(state)
end


function _record_outputs!(values, callbacks, observed::AbstractArray{<:Number}, t::Real, index::Integer)
    for k in eachindex(callbacks)
        value = callbacks[k](observed, t)
        value isa Number && isfinite(value) || throw(ArgumentError("outputs must return finite numeric scalars"))
        values[k][index] = value
    end
    
    return nothing
end


function _deterministic_evolution(model::_EvolutionModel, initial::Vector{ComplexF64}, callbacks,
                                  times::Vector{Float64}, options; state_view = identity, store_states::Bool = false)
                                  
    work = _propagation_workspace(model, options.method, options.krylov_dimension)
    state = copy(initial)
    outputs = _initial_outputs(callbacks, state_view(copy(state)), length(times))
    states = store_states ? [copy(state_view(state))] : nothing
    t = 0.0
    iterations = 0
    next_h = options.dt
    for save in 2:length(times)
        target = times[save]
        while t < target
            iterations += 1
            iterations <= options.max_steps || throw(ErrorException("maximum propagation steps exceeded"))
            proposed = min(next_h, options.dt)
            h = min(proposed, target - t)
            candidate, accepted_h, suggested = _accepted_step(model, state, t, h, work, options.atol, options.rtol)
            next_h = _next_step_size(proposed, h, accepted_h, suggested)
            copyto!(state, candidate)
            t = accepted_h == target - t ? target : t + accepted_h
        end
        
        observed = _observed_state(state, state_view, work)
        _record_outputs!(outputs, callbacks, observed, target, save)
        store_states && push!(states, copy(observed))
    end
    
    return (; times, outputs, final_state = copy(_observed_state(state, state_view, work)), states)
end
