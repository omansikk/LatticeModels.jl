# Bibliography:
#  for the constant N indexing:
#    Streltsov et al., 'General mapping for bosonic and fermionic operators in Fock space', doi:10.1103/PhysRevA.81.022124
#  for the constant N basis _next! function, although that is no longer in use
#    J. Zhang et al., 'Exact diagonalization: the bose–hubbard model as an example', doi:10.1088/0143-0807/31/3/016

import Base: getindex, length, iterate

# definitions of the basis types

abstract type Basis end

mutable struct Basis_constant_N <: Basis
    L::Integer
    N::Integer
end


mutable struct Basis_global_max_N <: Basis
    L::Integer
    N::Integer
end


mutable struct Basis_local_max_N <: Basis
    L::Integer
    N::Integer
end


mutable struct Basis_composite <: Basis
    bases::Vector{Basis}
    L::Int64
    N::Int64
end


Basis_composite(v::Vector{Basis}) = Basis_composite(v, sum(b.L for b in v), maximum(b.N for b in v))

# definitions the Basis objects as sort of abstract vectors, and allowing iteration

"""
    dimension(basis::Basis)

Compute the dimension, i.e., the number of basis vectors in `basis`.
"""
dimension(basis::Basis_constant_N) = _dimension_constant_N(basis.L, basis.N)
dimension(basis::Basis_global_max_N) = _dimension_global_max_N(basis.L, basis.N)
dimension(basis::Basis_local_max_N) = _dimension_local_max_N(basis.L, basis.N)


"""
    length(basis::Basis) = dimension(basis::Basis)

Compute the dimension, i.e., the number of basis vectors in `basis`.
"""
length(basis::Basis) = dimension(basis::Basis)


"""
    find_vector(basis::Basis, index::Integer)

Find the Fock vector corresponding to `index`.
"""
find_vector(basis::Basis_constant_N, index::Integer) = _find_vector_constant_N(basis.L, basis.N, index)
find_vector(basis::Basis_global_max_N, index::Integer) = _find_vector_global_max_N(basis.L, basis.N, index)
find_vector(basis::Basis_local_max_N, index::Integer) = _find_vector_local_max_N(basis.L, basis.N, index)


"""
    getindex(basis::Basis, index::Integer) = find_vector(basis::Basis, index::Integer)

Find the Fock vector corresponding to `index`.
"""
getindex(basis::Basis, index::Integer) = find_vector(basis, index)


function iterate(basis::Basis, index::Integer = 1)
    if index <= length(basis)
        return getindex(basis, index), index + 1
    else
        return nothing
    end
end


#functions for constant N bases

"""
    _next!(fock::AbstractVector{<:Integer})

Return the next Fock vector in order in the constant boson number basis. Probably useless now that the bases are iterable.
"""
function _next!(fock::AbstractVector{<:Integer})
    if length(fock) > 1
        N = sum(fock)
        nk = 1
        for j in reverse(1:length(fock) - 1)
            if fock[j] != 0
                nk = j
                break
            end
        end

        if fock[nk] > 0
            fock[nk] -= 1
            fock[nk + 1] = N - sum(@view fock[1:nk])
            fock[nk + 2:end] .*= 0
        else
            fock[:] .*= 0
            fock[1] = N
        end
    end

    return nothing
end


function _dimension_constant_N(L::Integer, N::Integer)
    d = 1.
    for i in 1:min(N, L - 1)
        d *= (L + N - i) / i
    end

    return round(Int64, d)
end


find_index(basis::Basis_constant_N, fock::AbstractVector{<:Integer}) = _find_index_constant_N(fock)

function _find_index_constant_N(fock::AbstractVector{<:Integer}, N::Integer = sum(fock))
    L = length(fock)
    index = 1
    prefix = 0
    for k in 1:L - 1
        prefix += fock[k]
        index += binomial(N + L - 1 - k - prefix, L - k)
    end

    return index
end


function _find_vector_constant_N(L::Integer, N::Integer, index::Integer)
    L == 1 && return Int64[N]
    fock = zeros(Int64, L)
    i = index; j = 0
    prefix = 0
    for l in 1:L - 1
        for n in 0:N
            fock[l] = n
            j = binomial(L + N - 1 - l - prefix - n, L - l)
            if i > j
                i -= j
                break
            end
        end
        prefix += fock[l]

        if l == L - 1
            fock[L] = N - prefix
        end

        if prefix == N
            break
        end
    end

    return fock
end


# functions for global max N bases

function _dimension_global_max_N(L::Integer, N::Integer)
    dim = 1
    for n in 1:N
        dim += _dimension_constant_N(L, n)
    end

    return dim
end


find_index(basis::Basis_global_max_N, fock::AbstractVector{<:Integer}) = _find_index_global_max_N(fock)

function _find_index_global_max_N(fock::AbstractVector{<:Integer})
    L = size(fock, 1)
    N = sum(fock)
    if N > 0
        return binomial(L + N - 1, L) + _find_index_constant_N(fock, N)
    else

        return 1
    end
end


function _find_vector_global_max_N(L::Integer, N::Integer, index::Integer)
    i = index
    for n in 0:N
        dim = _dimension_constant_N(L, n)
        if i - dim <= 0
            return _find_vector_constant_N(L, n, i)
        else
            i -= dim
        end
    end
end


# fuctions for local max N bases

function _dimension_local_max_N(L::Integer, N::Integer)
    return (N + 1) ^ L
end


find_index(basis::Basis_local_max_N, fock::AbstractVector{<:Integer}) = _find_index_local_max_N(basis.L, basis.N, fock)

function _find_index_local_max_N(L::Integer, N::Integer, fock::AbstractVector{<:Integer})
    index = 0
    radix = N + 1
    for l in 1:L
        index = radix * index + fock[l]
    end

    return index + 1
end


function _find_vector_local_max_N(L::Integer, N::Integer, index::Integer)
    radix = N + 1
    i = clamp(index - 1, 0, radix ^ L - 1)
    fock = zeros(Int64, L)
    for l in L:-1:1
        i, digit = divrem(i, radix)
        fock[l] = digit
    end

    return fock
end


#functions for composite bases

function dimension(basis::Basis_composite)
    dim = 1
    for b in basis.bases
        dim *= dimension(b)
    end
    
    return dim
end


function find_index(basis::Basis_composite, fock::AbstractVector{<:Integer})
    index = 0
    firstsite = 1
    for b in basis.bases
        lastsite = firstsite + b.L - 1
        index = index * length(b) + find_index(b, @view(fock[firstsite:lastsite])) - 1
        firstsite = lastsite + 1
    end

    return index + 1
end
    

function find_vector(basis::Basis_composite, index::Integer)
    sites = sum(b.L for b in basis.bases)
    fock = zeros(Int64, sites)
    remainder = index - 1
    lastsite = sites
    for b in Iterators.reverse(basis.bases)
        remainder, digit = divrem(remainder, length(b))
        firstsite = lastsite - b.L + 1
        fock[firstsite:lastsite] .= find_vector(b, digit + 1)
        lastsite = firstsite - 1
    end

    return fock
end


# functions for initialising product states

"""
    product_state(basis::Basis, fock::Vector{Int64})

    product_state(basis::Basis, fock::Tuple{Int64, Int64})

    product_state(basis::Basis, fock::Vector{Tuple{Int64, Int64}})

Return the product state `fock` represented in the basis `basis`.

E.g. the calls `product_state(basis, [0, 3, 0, 0])` and `product_state(basis, (3, 2))` return the same vector.

Similarly, `product_state(basis, [0, 3, 0, 2, 0])` and `product_state(basis, [(3, 2), (2, 4)])` result in the same vector.
"""
function product_state(basis::Basis, fock::Vector{Int64})
    state = zeros(ComplexF64, length(basis))
    state[find_index(basis, fock)] = 1.

    return state
end


function product_state(basis::Basis, fock::Tuple{Int64, Int64})
    state = zeros(ComplexF64, length(basis))
    vector = zeros(Int64, basis.L)
    vector[fock[2]] = fock[1]
    state[find_index(basis, vector)] = 1.

    return state
end


function product_state(basis::Basis, fock::Vector{Tuple{Int64, Int64}})
    state = zeros(ComplexF64, length(basis))
    vector = zeros(Int64, basis.L)
    for i in 1:length(fock)
        vector[fock[i][2]] = fock[i][1]
    end

    state[find_index(basis, vector)] = 1.

    return state
end


"""
    print_state(basis::Basis, state; cutoff = 0.99)

Print `state` prettily in descending order of the absolute value of the components.
"""
function print_state(basis::Basis, state::AbstractVector{<:Number}; cutoff::Real = 0.99)
    indices = sortperm(abs.(@view(state[1:length(basis)])), rev = true)
    total = 0.
    for i in 1:length(basis)
        total += abs2(state[indices[i]])
        label = "|" * prod(string.(basis[indices[i]])) * "⟩"
        println(round(state[indices[i]], digits = 4), " ", label, "         ", round(total, digits = 4))
        if total > cutoff
            break
        end
    end

    println()

    return nothing
end



