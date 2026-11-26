using Random

function _reference_mode_operator(basis, S)
    dim, L = length(basis), basis.L
    matrix = zeros(ComplexF64, dim, dim)
    for column in 1:dim
        input = basis[column]
        polynomial = Dict{Tuple,ComplexF64}(Tuple(zeros(Int, L)) => 1)
        for site in 1:L, particle in 1:input[site]
            expanded = Dict{Tuple,ComplexF64}()
            for (occupations, coefficient) in polynomial, mode in 1:L
                output = collect(occupations)
                output[mode] += 1
                key = Tuple(output)
                expanded[key] = get(expanded, key, 0.0im) + coefficient * conj(S[site, mode])
            end
            
            polynomial = expanded
        end
        
        for (occupations, coefficient) in polynomial
            normalization = sqrt(prod(factorial, occupations) / prod(factorial, input))
            matrix[find_index(basis, collect(occupations)), column] = coefficient * normalization
        end
    end
    
    return matrix
end

@testset "Direct bosonic mode transformations" begin
    rng = Xoshiro(1)
    for L in 1:4, N in 0:3, constructor in (Basis_constant_N, Basis_global_max_N)
        basis = constructor(L, N)
        state = 2.3 * randn(rng, ComplexF64, length(basis))
        original = copy(state)
        for periodic in (false, true)
            S = periodic ? [cispi(2l * k / L) / sqrt(L) for l in 1:L, k in 1:L] : [sqrt(2 / (L + 1)) * sinpi(l * k / (L + 1)) for l in 1:L, k in 1:L]
            T = transformation_operator(basis; periodic)
            direct = transform(basis, state; periodic)
            @test T ≈ _reference_mode_operator(basis, S) atol = 1e-11
            @test direct ≈ T * state atol = 1e-11
            @test transform(basis, direct; periodic, inverse = true) ≈ state atol = 1e-11
            @test T' * T ≈ I atol = 1e-11
            @test norm(direct) ≈ norm(state) atol = 1e-11
            @test state == original
            if !periodic
                @test eltype(T) == Float64
            end
        end
    end
end

@testset "Complex custom modes and transformation conventions" begin
    rng = Xoshiro(1)
    for L in 2:4, N in 1:3
        basis = Basis_constant_N(L, N)
        S = Matrix(qr(randn(rng, ComplexF64, L, L)).Q)
        original = copy(S)
        state = randn(rng, ComplexF64, length(basis))
        T = transformation_operator(basis; modes = S)
        inverse = transformation_operator(basis; modes = S, inverse = true)
        @test T ≈ _reference_mode_operator(basis, S) atol = 1e-11
        @test inverse ≈ T' atol = 1e-11
        @test transform(basis, state; modes = S) ≈ T * state atol = 1e-11
        @test transform(basis, T * state; modes = S, inverse = true) ≈ state atol = 1e-11
        @test S == original
        if N == 1
            @test T ≈ S' atol = 1e-12
        end
    end
    
    basis = Basis_constant_N(3, 2)
    S1 = Matrix(qr(randn(rng, ComplexF64, 3, 3)).Q)
    S2 = Matrix(qr(randn(rng, ComplexF64, 3, 3)).Q)
    @test transformation_operator(basis; modes = S1 * S2) ≈ transformation_operator(basis; modes = S2) * transformation_operator(basis; modes = S1) atol = 1e-11
end

@testset "Interference, sectors, and hopping modes" begin
    basis = Basis_constant_N(2, 2)
    balanced = [1.0 1; 1 -1] / sqrt(2)
    transformed = transform(basis, product_state(basis, [1, 1]); modes = balanced)
    @test transformed ≈ [1, 0, -1] / sqrt(2) atol = 1e-12
    @test transform(basis, zeros(3)) == zeros(3)

    many = Basis_constant_N(2, 30)
    actual = transform(many, product_state(many, [30, 0]); modes = balanced)
    expected = [sqrt(binomial(30, j)) / sqrt(2.0)^30 for j in 0:30]
    @test actual ≈ expected atol = 1e-10
    @test norm(actual) ≈ 1 atol = 1e-10

    for periodic in (false, true), L in 1:4, N in 0:3
        fixed = Basis_constant_N(L, N)
        T = transformation_operator(fixed; periodic)
        H = Matrix(hopping(fixed; periodic))
        diagonal = T * H * T'
        @test diagonal ≈ Diagonal(diag(diagonal)) atol = 1e-10
        energies = periodic ? [2cospi(2k / L) for k in 1:L] : [2cospi(k / (L + 1)) for k in 1:L]
        @test diag(diagonal) ≈ [sum(fixed[i] .* energies) for i in 1:length(fixed)] atol = 1e-10
    end
    
    global_basis = Basis_global_max_N(3, 3)
    for periodic in (false, true), N in 0:3
        fixed = Basis_constant_N(3, N)
        indices = [find_index(global_basis, fock) for fock in fixed]
        vector = zeros(ComplexF64, length(global_basis))
        vector[indices] .= 1:length(fixed)
        actual = transform(global_basis, vector; periodic)
        @test actual[indices] ≈ transform(fixed, vector[indices]; periodic) atol = 1e-11
        @test norm(actual[setdiff(1:length(vector), indices)]) < 1e-11
    end
    
    @test transform(Basis_global_max_N(3, 3), product_state(Basis_global_max_N(3, 3), [0, 0, 0]); periodic = true)[1] == 1
    for N in 0:5
        @test transformation_operator(Basis_constant_N(1, N); periodic = true) ≈ ones(1, 1)
    end
end
