@testset "Bases and product states" begin
    for L in 1:3, N in 0:3
        occupations = [collect(v) for v in Iterators.product(ntuple(_ -> 0:N, L)...)]
        for (basis, expected) in (
            (Basis_constant_N(L, N), filter(v -> sum(v) == N, vec(occupations))),
            (Basis_global_max_N(L, N), filter(v -> sum(v) <= N, vec(occupations))),
            (Basis_local_max_N(L, N), vec(occupations)),
        )
            @test length(basis) == dimension(basis) == length(expected)
            @test Set(Tuple(v) for v in basis) == Set(Tuple(v) for v in expected)
            @test all(find_index(basis, basis[i]) == i for i in 1:length(basis))
            @test all(find_vector(basis, i) == basis[i] for i in 1:length(basis))
        end
    end

    @test collect(Basis_constant_N(2, 2)) == [[2, 0], [1, 1], [0, 2]]
    @test collect(Basis_local_max_N(2, 1)) == [[0, 0], [0, 1], [1, 0], [1, 1]]

    parts = Basis[Basis_constant_N(2, 1), Basis_local_max_N(1, 2), Basis_global_max_N(2, 1)]
    composite = Basis_composite(parts)
    expected = [vcat(a, b, c) for a in parts[1] for b in parts[2] for c in parts[3]]
    @test composite.L == 5
    @test length(composite) == 18
    @test collect(composite) == expected
    @test all(find_index(composite, v) == i for (i, v) in enumerate(expected))

    basis = Basis_constant_N(3, 2)
    state = product_state(basis, [0, 2, 0])
    @test state == ComplexF64[0, 0, 0, 1, 0, 0]
    @test state == product_state(basis, (2, 2))
    @test product_state(basis, [1, 0, 1]) == product_state(basis, [(1, 1), (1, 3)])
    @test norm(state) == 1
end
