@testset "Observables" begin
    for basis in (Basis_constant_N(2, 1), Basis_global_max_N(2, 1), Basis_local_max_N(2, 1))
        left = product_state(basis, [1, 0])
        right = product_state(basis, [0, 1])
        bell = (left + 1im * right) / sqrt(2)
        @test entropy(basis, left) ≈ 0 atol = 1e-14
        @test entropy(basis, bell) ≈ log(2)
        @test entropy(basis, bell; sites = 0) ≈ 0 atol = 1e-14
        @test entropy(basis, bell; sites = 2) ≈ 0 atol = 1e-14
        @test_throws ArgumentError entropy(basis, left; sites = 3)
        @test_throws DimensionMismatch entropy(basis, zeros(length(basis) + 1))
        @test inverse_participation_ratio(left, numbers(basis)) ≈ 0 atol = 1e-14
        @test inverse_participation_ratio(bell, numbers(basis)) ≈ 1
    end
    
    vacuum_basis = Basis_global_max_N(2, 1)
    @test inverse_participation_ratio(product_state(vacuum_basis, [0, 0]), numbers(vacuum_basis)) == 0
    single = Basis_constant_N(1, 2)
    @test inverse_participation_ratio(product_state(single, [2]), numbers(single)) == 0

    H = [1.0 0.5; 0.5 -2.0]
    state = [1.0, 0.0]
    @test moments(H, state, 3) ≈ permutedims(hcat(H * state, H^2 * state, H^3 * state))
end

@testset "Open-boundary basis transform" begin
    for N in 1:2
        basis = Basis_constant_N(3, N)
        T = transformation_operator(basis)
        @test T' * T ≈ I atol = 1e-12
        transformed = T' * hopping(basis) * T
        @test transformed ≈ Diagonal(diag(transformed)) atol = 1e-12
    end
end

@testset "Local entropy versus reduced density matrix" begin
    basis = Basis_local_max_N(4, 2)
    state = normalize!(ComplexF64[sin(i) + im * cos(2i) for i in 1:length(basis)])
    original = copy(state)
    focks = collect(basis)
    for sites in 0:basis.L
        rows = [foldl((index, n) -> 3index + n, fock[1:sites]; init = 0) + 1 for fock in focks]
        reduced = zeros(ComplexF64, 3^sites, 3^sites)
        for i in eachindex(state), j in eachindex(state)
            if @view(focks[i][sites+1:end]) == @view(focks[j][sites+1:end])
                reduced[rows[i], rows[j]] += state[i] * conj(state[j])
            end
        end
        
        eigenvalues = eigvals(Hermitian(reduced))
        expected = -sum(p * log(p) for p in eigenvalues if p > 1e-14)
        @test entropy(basis, state; sites) ≈ expected atol = 1e-12
        @test entropy(basis, sparse(state); sites) ≈ expected atol = 1e-12
        @test state == original
    end
end
