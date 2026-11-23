using LinearAlgebra


struct _ModeRotation{T<:Number}
    p::Int
    q::Int
    matrix::Matrix{T}
    blocks::Dict{Int,Matrix{T}}
end


struct _ModeTransformPlan{T<:Number}
    rotations::Vector{_ModeRotation{T}}
    phases::Vector{T}
end


function _single_particle_modes(L::Integer, periodic::Bool)
    if periodic
        return [cispi(2 * l * k / L) / sqrt(L) for l in 1:L, k in 1:L]
    end
    
    return [sqrt(2 / (L + 1)) * sinpi(l * k / (L + 1)) for l in 1:L, k in 1:L]
end


function _mode_transform_plan(basis::Union{Basis_constant_N,Basis_global_max_N}; periodic::Bool = false,
                              modes::Union{Nothing,AbstractMatrix{<:Number}} = nothing, inverse::Bool = false)
                              
    S = modes === nothing ? _single_particle_modes(basis.L, periodic) : modes
    T = eltype(S) <: Real ? Float64 : ComplexF64
    A = Matrix{T}(inverse ? S : S')
    rotations = _ModeRotation{T}[]
    rowp, rowq = zeros(T, basis.L), zeros(T, basis.L)
    for j in 1:basis.L-1, i in basis.L:-1:j+1
        p, q = i - 1, i
        x, y = A[p, j], A[q, j]
        r = hypot(abs(x), abs(y))
        (iszero(r) || abs(y) <= eps(Float64) * r) && continue
        g = T[conj(x)/r conj(y)/r; -y/r x/r]
        copyto!(rowp, @view(A[p, :]))
        copyto!(rowq, @view(A[q, :]))
        @views A[p, :] .= g[1, 1] .* rowp .+ g[1, 2] .* rowq
        @views A[q, :] .= g[2, 1] .* rowp .+ g[2, 2] .* rowq
        push!(rotations, _ModeRotation(p, q, Matrix(g'), Dict{Int,Matrix{T}}()))
    end
    
    return _ModeTransformPlan(rotations, collect(diag(A)))
end


function _two_mode_block(g::AbstractMatrix{T}, particles::Integer) where {T<:Number}
    previous = ones(T, 1, 1)
    for m in 1:particles
        block = zeros(T, m + 1, m + 1)
        for j in 0:m
            source = j == m ? 2 : 1
            column = min(j, m - 1) + 1
            normalization = sqrt(j == m ? m : m - j)
            for k in 0:m-1
                amplitude = previous[k + 1, column] / normalization
                block[k + 1, j + 1] += g[1, source] * sqrt(m - k) * amplitude
                block[k + 2, j + 1] += g[2, source] * sqrt(k + 1) * amplitude
            end
        end
        
        previous = block
    end
    
    return previous
end


function _apply_mode_plan!(state::AbstractVector{<:Number}, basis::Union{Basis_constant_N,Basis_global_max_N},
                           plan::_ModeTransformPlan, indices::Vector{Int}, buffer::AbstractVector{<:Number})
                           
    for i in 1:length(basis)
        fock = find_vector(basis, i)
        phase = one(eltype(plan.phases))
        for l in eachindex(fock)
            phase *= plan.phases[l]^fock[l]
        end
        
        state[i] *= phase
    end
    
    for rotation in Iterators.reverse(plan.rotations)
        p, q = rotation.p, rotation.q
        for i in 1:length(basis)
            fock = find_vector(basis, i)
            fock[q] == 0 || continue 
            m = fock[p]
            m == 0 && continue
            for j in 0:m
                fock[p], fock[q] = m - j, j
                indices[j + 1] = find_index(basis, fock)
                buffer[j + 1] = state[indices[j + 1]]
            end
            
            all(iszero, @view(buffer[1:m+1])) && continue
            block = get!(rotation.blocks, m) do
                _two_mode_block(rotation.matrix, m)
            end
            
            for row in 1:m+1
                value = zero(eltype(state))
                for column in 1:m+1
                    value += block[row, column] * buffer[column]
                end
                
                state[indices[row]] = value
            end
        end
    end
    
    return state
end


"""
    transform(basis, state; periodic = false, modes = nothing, inverse = false)

Supports only `Basis_constant_N` and `Basis_global_max_N`.
"""
function transform(basis::Union{Basis_constant_N,Basis_global_max_N}, state::AbstractVector{<:Number};
                   periodic::Bool = false, modes::Union{Nothing,AbstractMatrix{<:Number}} = nothing, inverse::Bool = false)
                   
    plan = _mode_transform_plan(basis; periodic, modes, inverse)
    T = promote_type(eltype(state), eltype(plan.phases), Float64)
    result = Vector{T}(state)
    indices = zeros(Int, basis.N + 1)
    buffer = zeros(T, basis.N + 1)
    
    return _apply_mode_plan!(result, basis, plan, indices, buffer)
end


