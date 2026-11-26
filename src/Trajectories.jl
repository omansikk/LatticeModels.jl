# Quantum-jump trajectories: Daley, Adv. Phys. 63, 77 (2014).

using Distributed, Random

struct _TrajectoryModel{H,J,G,D,B} <: _EvolutionModel
    hamiltonian::H
    jumps::J
    generator::G
    decay::D
    buffer::B
    dimension::Int
end


function _effective_generator(H::AbstractMatrix{<:Number}, jumps)
    generator = -1im * H
    for C in jumps
        generator .-= 0.5 .* (C' * C)
    end
    
    return generator
end


function _trajectory_model(H, jumps, dim::Integer)
    H0 = _checked_hamiltonian(H isa AbstractMatrix ? H : H(0.0), dim)
    dynamic_jumps = applicable(jumps, 0.0)
    C0 = _checked_jumps(dynamic_jumps ? jumps(0.0) : jumps, dim)
    Cs = dynamic_jumps ? jumps : C0
    generator = H isa AbstractMatrix && !dynamic_jumps ? _effective_generator(H0, C0) : nothing
    decay = !(H isa AbstractMatrix) && !dynamic_jumps && !isempty(C0) ? _effective_generator(spzeros(ComplexF64, dim, dim), C0) : nothing
    
    return _TrajectoryModel(H, Cs, generator, decay, H isa AbstractMatrix ? nothing : _generator_buffer(H0), dim)
end


_jump_operators(model::_TrajectoryModel, t::Real) = applicable(model.jumps, t) ? _checked_jumps(model.jumps(t), model.dimension) : model.jumps

function _generator(model::_TrajectoryModel, t::Real)
    model.generator !== nothing && return model.generator
    H = model.hamiltonian isa AbstractMatrix ? model.hamiltonian : model.hamiltonian(t)
    H = _checked_hamiltonian(H, model.dimension)
    model.decay !== nothing && return _reuse_generator!(model.buffer, H, -1im, model.decay)
    
    if !applicable(model.jumps, t) && isempty(model.jumps)
        return _reuse_generator!(model.buffer, H, -1im, nothing)
    end
    
    return _effective_generator(H, _jump_operators(model, t))
end


function _locate_jump(model::_TrajectoryModel, state::Vector{ComplexF64}, candidate::Vector{ComplexF64}, 
                      t::Float64, h::Float64, threshold::Float64, work::_Propagation, jump_tol::Real)
                      
    lo, hi = 0.0, h
    event_state = copyto!(work.steps.event, candidate)
    for iteration in 1:80
        hi - lo <= jump_tol * max(1.0, abs(t + hi)) && break
        midpoint = (lo + hi) / 2
        trial = _two_half_steps!(work.steps.trial, model, state, t, midpoint, work)
        if sum(abs2, trial) <= threshold
            hi = midpoint
            copyto!(event_state, trial)
        else
            lo = midpoint
        end
    end
    
    return event_state, hi
end


function _apply_jump(state::Vector{ComplexF64}, jumps, rng::AbstractRNG)
    weights = [sum(abs2, C * state) for C in jumps]
    total = sum(weights)
    total > 0 || throw(ErrorException("a norm crossing occurred with no active jump channel"))
    draw = rand(rng) * total
    chosen = findlast(>(0), weights)
    for k in eachindex(weights)
        draw -= weights[k]
        if draw < 0
            chosen = k
            break
        end
    end
    
    return normalize!(Vector{ComplexF64}(jumps[chosen] * state))
end


function _trajectory(H, jumps, initial::Vector{ComplexF64}, callbacks, times::Vector{Float64}, options, rng::AbstractRNG)
    model = _trajectory_model(H, jumps, length(initial))
    has_jumps = applicable(model.jumps, 0.0) || !isempty(model.jumps)
    if !has_jumps
        return _deterministic_evolution(model, initial, callbacks, times, options;
            state_view = normalize).outputs
    end
    
    work = _propagation_workspace(model, options.method, options.krylov_dimension)
    state = copy(initial)
    outputs = _initial_outputs(callbacks, copy(state), length(times))
    threshold = rand(rng)
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
            candidate, accepted_h, suggested = _accepted_step(model, state, t, h, work, options.atol, options.rtol; contractive = true)
            next_h = _next_step_size(proposed, h, accepted_h, suggested)
            if has_jumps && sum(abs2, candidate) <= threshold
                candidate, event_h = _locate_jump(model, state, candidate, t, accepted_h, threshold, work, options.jump_tol)
                state .= _apply_jump(candidate, _jump_operators(model, t + event_h), rng)
                t += event_h
                next_h = min(options.dt, accepted_h)
                threshold = rand(rng)
            else
                state .= candidate
                has_jumps || normalize!(state)
                t = accepted_h == target - t ? target : t + accepted_h
            end
        end
        
        observed = _observed_state(state, normalize, work)
        _record_outputs!(outputs, callbacks, observed, target, save)
    end
    
    return outputs
end


function _accumulate_output!(mean::Vector{<:Number}, deviation::Vector{<:Real}, value::Vector{<:Number}, count::Integer)
    axes(mean) == axes(deviation) == axes(value) || throw(DimensionMismatch("output arrays must have matching axes"))
    @inbounds @simd for i in eachindex(mean)
        delta = value[i] - mean[i]
        mean[i] += delta / count
        deviation[i] += real(conj(delta) * (value[i] - mean[i]))
    end
    
    return nothing
end


function _accumulate_outputs!(means, deviations, values, count::Integer)
    axes(means) == axes(deviations) == axes(values) || throw(DimensionMismatch("output collections must have matching axes"))
    for k in eachindex(means)
        _accumulate_output!(means[k], deviations[k], values[k], count)
    end
    
    return nothing
end


"""
    run_trajectories(basis, H, initial_state, outputs, end_time;
                     jump_operators = (), saveat = nothing, saves = 1000, dt = nothing,
                     steps = 0, trajectories = 1, method = :auto, krylov_dimension = 0,
                     atol = 1e-10, rtol = 1e-8, jump_tol = 1e-10,
                     rng = Random.default_rng(), parallel = :serial, realization = nothing,
                     max_steps = 10^7)

`H` is a Hermitian matrix or a callable `H(t)`
.
`jump_operators` is a collection of matrices or a callable `jumps(t)`.

`:auto` uses dense exponentials for dense generators and Arnoldi exponential actions for sparse generators. 
"""
function run_trajectories(basis::Basis, H, initial_state::AbstractVector{<:Number}, outputs, end_time::Real;
                          jump_operators = (), saveat = nothing, saves::Integer = 1000, dt::Union{Nothing,Real} = nothing,
                          steps::Integer = 0, trajectories::Integer = 1, method::Symbol = :auto, krylov_dimension::Integer = 0,
                          atol::Real = 1e-10, rtol::Real = 1e-8, jump_tol::Real = 1e-10, rng::AbstractRNG = Random.default_rng(), 
                          parallel::Symbol = :serial, realization = nothing, max_steps::Integer = 10^7)
    
    times = _save_times(end_time, saveat, saves)
    initial = _pure_state(basis, initial_state)
    options = _solver_options(end_time, times; dt, steps, method, krylov_dimension, atol, rtol, max_steps)
    parallel in (:serial, :distributed) || throw(ArgumentError("parallel must be :serial or :distributed")) #maybe add GPU
    callbacks = _output_callbacks(outputs, initial)
    options = (; options..., jump_tol)
    seeds = rand(rng, UInt64, trajectories)
    run_one = function (index)
        local_rng = Xoshiro(seeds[index])
        problem = realization === nothing ? (; hamiltonian = H, jump_operators) : realization(local_rng)
        
        return _trajectory(problem.hamiltonian, problem.jump_operators, initial, callbacks, times, options, local_rng)
    end
    
    means = nothing
    deviations = nothing
    count = 0
    batch_size = parallel == :serial ? 1 : max(1, 2nworkers())
    for start in 1:batch_size:trajectories
        indices = start:min(start + batch_size - 1, trajectories)
        batch = parallel == :serial ? map(run_one, indices) : pmap(run_one, indices)
        for values in batch
            count += 1
            if means === nothing
                means = [copy(v) for v in values]
                deviations = [zeros(Float64, length(v)) for v in values]
            else
                _accumulate_outputs!(means, deviations, values, count)
            end
        end
    end
    
    standard_errors = [count == 1 ? fill(NaN, length(v)) : sqrt.(max.(v, 0) ./ (count * (count - 1))) for v in deviations]
    
    return (; times, outputs = means, standard_errors)
end
