# Bibliography:
# Peotta et al., 'Determination of dynamical quantum phase transitions in strongly correlated many-body systems using Loschmidt cumulants', doi:10.1103/PhysRevX.11.041018

using LinearAlgebra

"""
    inverse_participation_ratio(state, numbers)

Compute the scaled inverse participation ratio from the mean site occupations.
The total mean occupation provides the normalization, independently of basis
ordering. Return zero for a vacuum state or a single-site system.
"""
function inverse_participation_ratio(state::AbstractVector{<:Number}, ns::AbstractVector{<:AbstractMatrix})
    L = size(ns, 1)
    occupations = [real(dot(state, n, state)) for n in ns]
    out = sum(abs2, occupations)
    (L == 1 || iszero(out)) && return 0.0
    
    return (sum(occupations) ^ 2 / out - 1.) / (L - 1)
end


function _compute_entropy(schmidt::AbstractMatrix{<:Number})
    S = svdvals(schmidt)
    entropy = 0.0
    for s in S
        alpha2 = abs2(s)
        if alpha2 > 0
            entropy += -alpha2 * log(alpha2)
        end
    end

    return entropy
end


"""
    entropy(basis::Basis, state; sites = cld(basis.L, 2))

Compute the bipartite von Neumann entanglement entropy of a normalized pure
state, with natural logarithms. The first `sites` lattice sites form one
subsystem; the default is half a chain, rounded up.
"""
function entropy(basis::Union{Basis_constant_N, Basis_global_max_N}, state::AbstractVector{<:Number}; sites::Integer = cld(basis.L, 2))
    sub_dim = sites == 0 ? 1 : _dimension_global_max_N(sites, basis.N)
    rest_dim = sites == basis.L ? 1 : _dimension_global_max_N(basis.L - sites, basis.N)
    schmidt = zeros(ComplexF64, sub_dim, rest_dim)
    for i in 1:length(basis)
        fock = basis[i]
        schmidt[_find_index_global_max_N(@view(fock[1:sites])), _find_index_global_max_N(@view(fock[sites + 1:end]))] = state[i]
    end

    return _compute_entropy(schmidt)
end


function entropy(basis::Basis_local_max_N, state::AbstractVector{<:Number}; sites::Integer = cld(basis.L, 2))
    sub_dim = _dimension_local_max_N(sites, basis.N)
    rest_dim = _dimension_local_max_N(basis.L - sites, basis.N)
    schmidt = reshape(Vector{ComplexF64}(state), rest_dim, sub_dim)
    
    return _compute_entropy(schmidt)
end


function _toeplitz(vals::AbstractVector{<:Number})
    dim = (length(vals) + 1) ÷ 2
    T = zeros(ComplexF64, dim, dim)
    center = length(vals) - dim + 1
    for i in 1:dim, j in 1:dim
        T[i, j] = vals[center + i - j]
    end

    return T
end


function moments(H::AbstractMatrix{<:Number}, state::AbstractVector{<:Number}, n::Integer)
    dim = length(state)
    V = Matrix{Float64}(undef, n, dim)
    mul!(@view(V[1, :]), H, state)
    for i in 2:n
        mul!(@view(V[i, :]), H, @view(V[i - 1, :]))
    end

    return V
end


"""
    loschmidt_zeros(moments, nlz, time, eps = 1e-2)`

Compute the `nlz` Loschmidt zeros closest to `time`.
"""
function loschmidt_zeros(moments::AbstractVector{<:Number}, nlz::Integer, time::Number, eps::Real = 1e-2)
    cs = copy(moments)
    nc = length(moments)
    for n in 2:nc
        for m in 1:n - 1
            cs[n] -= binomial(n - 1, m - 1) * cs[m] * moments[n - m]
        end
    end

    denominator = big(1)
    for n in 1:nc
        cs[n] *= (-1)^(n - 1) / denominator
        denominator *= n
    end

    ks = @view cs[end - nlz + 1:end]
    T = _toeplitz(@view cs[nc - 2 * nlz:nc - 1])
    as = T \ ks

    h = zeros(ComplexF64, nlz, nlz)
    h[1, :] .= as
    for i in 2:nlz
        h[i, i - 1] = 1
    end
    
    lambdas = eigvals(h)
    V = zeros(ComplexF64, nlz, nlz)
    for i in 1:nlz
        V[i, :] .= lambdas .^ (i - 1)
    end

    ds = V \ ks ./ lambdas .^ (nc - nlz + 1)
    out::Array{ComplexF64, 1} = []
    for i in 1:nlz
        if abs(1 - real(ds[i])) < eps
            push!(out, time - 1im / lambdas[i])
        end
    end

    return out
end
