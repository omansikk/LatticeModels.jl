using SparseArrays, LinearAlgebra


"""
    operator(basis::Basis, action::Function; element_type = Float64, add_adjoint = false)

Construct a sparse matrix representation (`SparseMatrixCSC{element_type, Int64}`) of an operator whose action on the basis vectors `basis[i]` is defined by the function `action(fock)`. The function `action` should take as its argument a basis vector and return one or more basis vectors and corresponding matrix elements:

`fock, element = action(basis[i])`,

where `typeof(fock)` is `Vector{Int64}` or `Vector{Vector{Int64}}` and `typeof(element)` is `element_type` or `Vector{element_type}`.

Note that the function assumes that the output of `action` actually belongs to `basis`. If it does not, the function either breaks or works incorrectly.
"""
function operator(basis::Basis, action::Function; element_type::Type{<:Number} = Float64, add_adjoint::Bool = false)
    dim = length(basis)
    rows, columns, elements = Int[], Int[], element_type[]
    sizehint!(rows, dim); sizehint!(columns, dim); sizehint!(elements, dim)
    for i in 1:dim
        fock, element = action(basis[i])
        if length(fock) != 0
            if isa(fock[1], Vector)
                if promote_type(element_type, eltype(element)) != element_type
                    column = Dict{Int,element_type}()
                    for j in eachindex(fock)
                        if element[j] != 0
                            row = find_index(basis, fock[j])
                            column[row] = get(column, row, zero(element_type)) + element[j]
                        end
                    end
                    
                    for (row, value) in column
                        push!(rows, row); push!(columns, i); push!(elements, value)
                    end
                    
                    continue
                end
                
                for j in 1:length(fock)
                    if element[j] != 0
                        push!(rows, find_index(basis, fock[j]))
                        push!(columns, i)
                        push!(elements, element[j])
                    end
                end
                
            elseif element != 0
                push!(rows, find_index(basis, fock))
                push!(columns, i)
                push!(elements, element)
            end
        end
    end

    op = sparse(rows, columns, elements, dim, dim)
    if add_adjoint == true
        op .+= op'
    end

    return op
end


_sparse_diagonal(basis::Basis, action::F) where {F} = dropzeros!(spdiagm(0 => diagonal_operator(basis, action)))

"""
    operator(basis::Basis, action::Function; element_type = Float64, add_adjoint = false)

Construct a vector representation of a diagonal operator whose action on the basis vectors `basis[i]` is defined by the function `action(fock)`. The function `action` should take as its argument a basis vector and return the corresponding matrix element:

element = action(basis[i])`,

where `typeof(element)` is `element_type`.
"""
function diagonal_operator(basis::Basis, action::Function; element_type::Type{<:Number} = Float64)
    dim = length(basis)
    op = zeros(element_type, dim)
    for i in 1:dim
        op[i] += action(basis[i])
    end

    return op
end


"""
    number(basis::Basis, site::Int64)

Construct the number operator ``\\hat{n}_\\ell`` on the site `site` in the basis `basis`.
"""
function number(basis::Basis, site::Int64)
    return _sparse_diagonal(basis, (fock::AbstractVector{<:Integer}) -> fock[site])
end


"""
    numbers(basis::Basis)

Create a vector of all the local number operators in the basis `basis`. See `number(basis, site)`
"""
function numbers(basis::Basis)
    return [number(basis, l) for l in 1:basis.L]
end


"""
    annihilation(basis::Basis, site::Int64)

Construct the annihilation operator ``\\hat{a}_\\ell`` on the site `site` in the basis `basis`.
"""
function annihilation(basis::Basis, site::Int64)
    return operator(basis, (fock::AbstractVector{<:Integer}) ->
        begin
            if fock[site] > 0
                fock_out = copy(fock)
                fock_out[site] -= 1

                return fock_out, sqrt(fock[site])
            else
                return fock, 0
            end
        end
    )
end


"""
    annihilations(basis::Basis, site::Int64)

Create a vector of all the local annihilation operators in the basis `basis`. See `annihilation(basis, site)`
"""
function annihilations(basis::Basis)
    return [annihilation(basis, l) for l in 1:basis.L]
end


function _hops(basis::Basis, fock::AbstractVector{<:Integer}; periodic::Integer = false)
    focks_out::Vector{Vector{Int64}} = []
    elements::Vector{Float64} = []
    copy_fock = copy(fock)
    for l in 1:basis.L - 1
        if fock[l] > 0 && (!(basis isa Basis_local_max_N) || fock[l + 1] < basis.N)
            copyto!(copy_fock, fock)
            copy_fock[l] -= 1
            copy_fock[l + 1] += 1
            if find_index(basis, copy_fock) <= length(basis)
                push!(focks_out, copy(copy_fock))
                push!(elements, sqrt(fock[l] * copy_fock[l + 1]))
            end
        end

    end

    if periodic == true && fock[basis.L] > 0 && (!(basis isa Basis_local_max_N) || fock[1] < basis.N)
        copyto!(copy_fock, fock)
        copy_fock[basis.L] -= 1
        copy_fock[1] += 1
        if find_index(basis, copy_fock) <= length(basis)
            push!(focks_out, copy(copy_fock))
            push!(elements, sqrt(fock[basis.L] * copy_fock[1]))
        end
    end

    return focks_out, elements
end


function _hops_2D(basis::Basis, fock::AbstractVector{<:Integer}, L1::Integer, L2::Integer; periodic::Integer = 0)
    L1 > 0 && L2 > 0 && L1 * L2 == basis.L || throw(ArgumentError("positive dimensions must multiply to basis.L"))
    periodic in 0:2 || throw(ArgumentError("periodic must be 0, 1, or 2"))
    focks_out::Vector{Vector{Int64}} = []
    elements::Vector{Float64} = []
    for k in 1:L2, j in 1:L1
        site = j + (k - 1) * L1
        fock[site] == 0 && continue
        right = j < L1 ? site + 1 : (periodic >= 1 && L1 > 1 ? site - L1 + 1 : 0)
        above = k < L2 ? site + L1 : (periodic == 2 && L2 > 1 ? j : 0)
        for destination in (right, above)
            destination == 0 && continue
            basis isa Basis_local_max_N && fock[destination] >= basis.N && continue
            copy_fock = copy(fock)
            copy_fock[site] -= 1
            copy_fock[destination] += 1
            if find_index(basis, copy_fock) <= length(basis)
                push!(focks_out, copy_fock)
                push!(elements, sqrt(fock[site] * copy_fock[destination]))
            end
        end
    end

    return focks_out, elements
end


function hopping(basis::Basis; periodic::Integer = false, J::Number = 1, dims::Tuple{Int64, Int64} = (0, 0))
    if dims == (0, 0)
        HJ = operator(basis, (fock::AbstractVector{<:Integer}) -> _hops(basis, fock, periodic = periodic), add_adjoint = true) .* J
    else
        if periodic == false
            HJ = operator(basis, (fock::AbstractVector{<:Integer}) -> _hops_2D(basis, fock, dims[1], dims[2], periodic = 0), add_adjoint = true) .* J
        else
            HJ = operator(basis, (fock::AbstractVector{<:Integer}) -> _hops_2D(basis, fock, dims[1], dims[2], periodic = periodic), add_adjoint = true) .* J
        end
    end
    
    return HJ
end


_u(fock::AbstractVector{<:Integer}) = -0.5 * sum(n -> n^2 - n, fock; init = 0)

function anharmonicity(basis::Basis; U::Number = 1)
    return _sparse_diagonal(basis, _u) .* U
end


"""
    ABH_Hamiltonian(basis::Basis; periodic = false, split = true, J = 1, U = 1, dims = (0, 0))

Construct a sparse matrix representation of the attractive Bose-Hubbard Hamiltonian
``
\\hat{H} = -\\frac{U}{2}\\sum_\\ell \\hat{n} (\\hat{n}_\\ell - 1) + J \\sum_\\ell (\\hat{a}^\\dagger_\\ell \\hat{a}_{\\ell + 1} + \\text{H.c.}).
``

The interaction and hopping terms are returned separately by default (`split = true`).

For a rectangular 2D array, set `dims = (L1, L2)`, where `L1` and `L2` are the lengths of the edges. Set `periodic` to 1 for a periodic array in one direction, and 2 for both directions.
"""
function ABH_Hamiltonian(basis::Basis; periodic::Integer = false, split::Bool = true, J::Number = 1, U::Number = 1, dims::Tuple{Int64, Int64} = (0, 0))
    HU = anharmonicity(basis, U = U)
    HJ = hopping(basis::Basis; periodic = periodic, J = J, dims = dims)
    if split == true
        return HU, HJ
    else
        return HU .+ HJ
    end
end



"""
    number_disorder(basis::Basis; pattern = 2. .* rand(basis.L) .- 1.)

Construct a sparse matrix representation of on-site disorder
``
\\hat{H}_{\\omega} = \\sum_\\ell \\omega_\\ell \\hat{n}_\\ell.
``
"""
function number_disorder(basis::Basis; pattern::AbstractVector{<:Number} = 2. .* rand(basis.L) .- 1.)
    return _sparse_diagonal(basis, (fock::AbstractVector{<:Integer}) -> _weighted_sum(pattern, fock, identity))
end


"""
    anharmonicity_disorder(basis::Basis; pattern = 2. .* rand(basis.L) .- 1.)

Construct a sparse matrix representation of on-site anharmonicity disorder
``
\\hat{H}_{\\delta U} = -\\frac{1}{2}\\sum_\\ell \\delta U_\\ell \\hat{n}_\\ell (\\hat{n}_\\ell - 1).
``
"""
function anharmonicity_disorder(basis::Basis; pattern::AbstractVector{<:Number} = 2. .* rand(basis.L) .- 1)
    return _sparse_diagonal(basis, (fock::AbstractVector{<:Integer}) -> -0.5 * _weighted_sum(pattern, fock, n -> n^2 - n))
end


function _weighted_sum(pattern::AbstractVector, fock::AbstractVector, f::F) where {F}
    !isempty(fock) && axes(pattern) == axes(fock) &&
        return sum(i -> pattern[i] * f(fock[i]), eachindex(fock))
        
    return sum(pattern .* f.(fock))
end


function _vectorize(M::AbstractMatrix{<:Number})
    return Vector{eltype(M)}(vec(M))
end


"""
    transformation_operator(basis; periodic = false, modes = nothing, inverse = false)

Build the explicit dense many-body site-to-mode transformation matrix. Supports `Basis_constant_N` and `Basis_global_max_N`.
"""
function transformation_operator(basis::Union{Basis_constant_N,Basis_global_max_N}; periodic::Bool = false, modes::Union{Nothing,AbstractMatrix{<:Number}} = nothing, inverse::Bool = false)
    plan = _mode_transform_plan(basis; periodic, modes, inverse)
    T = eltype(plan.phases)
    dim = length(basis)
    matrix = zeros(T, dim, dim)
    indices = zeros(Int, basis.N + 1)
    buffer = zeros(T, basis.N + 1)
    for column in 1:dim
        matrix[column, column] = one(T)
        _apply_mode_plan!(@view(matrix[:, column]), basis, plan, indices, buffer)
    end
    
    return matrix
end


function transformation_operator(basis::Basis; kwargs...)
    throw(ArgumentError("mode transforms support only Basis_constant_N and Basis_global_max_N"))
end
