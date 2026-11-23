struct _UnitaryModel{H,G} <: _EvolutionModel
    hamiltonian::H
    generator::G
    dimension::Int
end


function _unitary_model(H, dim::Integer)
    H0 = _checked_hamiltonian(H isa AbstractMatrix ? H : H(0.0), dim)
    
    return _UnitaryModel(H, H isa AbstractMatrix ? -1im * H0 : nothing, dim)
end


function _generator(model::_UnitaryModel, t::Real)
    model.generator !== nothing && return model.generator
    
    return -1im * _checked_hamiltonian(model.hamiltonian(t), model.dimension)
end


"""
    run_unitary(basis, H, initial_state, outputs, end_time;
                saveat = nothing, saves = 1000, dt = nothing, steps = 0, method = :auto,
                krylov_dimension = 0, atol = 1e-10, rtol = 1e-8,
                max_steps = 10^7, store_states = false)

Coherent pure-state evolution with a Hermitian matrix `H` or `H(t)`. Outputs accepts functions `(psi, t)` or `(psi)`.

Return `(times, outputs, final_state, states)`. `outputs[k][i]` is output `k` at `times[i]`. 
"""
function run_unitary(basis::Basis, H, initial_state::AbstractVector{<:Number}, outputs, end_time::Real;
                     saveat = nothing, saves::Integer = 1000, dt::Union{Nothing,Real} = nothing, steps::Integer = 0,
                     method::Symbol = :auto, krylov_dimension::Integer = 0, atol::Real = 1e-10, rtol::Real = 1e-8,
                     max_steps::Integer = 10^7, store_states::Bool = false)
                     
    times = _save_times(end_time, saveat, saves)
    initial = _pure_state(basis, initial_state)
    options = _solver_options(end_time, times; dt, steps, method, krylov_dimension, atol, rtol, max_steps)
    callbacks = _output_callbacks(outputs, initial)
    
    return _deterministic_evolution(_unitary_model(H, length(initial)), initial, callbacks, times, options; state_view = normalize, store_states)
end
