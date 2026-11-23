@testset "Operators and Hamiltonians" begin
    for basis in (Basis_global_max_N(2, 2), Basis_local_max_N(2, 2))
        ns, as = numbers(basis), annihilations(basis)
        vacuum = product_state(basis, [0, 0])
        for site in 1:2
            @test ns[site] isa SparseMatrixCSC
            @test ns[site] == number(basis, site)
            @test as[site] == annihilation(basis, site)
            @test as[site]' * as[site] ≈ ns[site]
            @test iszero(as[site] * vacuum)
        end
        
        @test as[1] * product_state(basis, [2, 0]) ≈ sqrt(2) * product_state(basis, [1, 0])
    end

    basis = Basis_constant_N(2, 2) 
    expected_hop = [0 sqrt(2) 0; sqrt(2) 0 sqrt(2); 0 sqrt(2) 0]
    @test hopping(basis) ≈ expected_hop
    @test hopping(basis; periodic = true) ≈ 2expected_hop
    @test anharmonicity(basis; U = 3) == Diagonal([-3, 0, -3])
    HU, HJ = ABH_Hamiltonian(basis; J = 0.7, U = 3)
    @test HU == Diagonal([-3, 0, -3])
    @test HJ ≈ 0.7expected_hop
    @test ABH_Hamiltonian(basis; J = 0.7, U = 3, split = false) ≈ HU + HJ
    @test ishermitian(HU + HJ)
    @test number_disorder(basis; pattern = [2, -1]) == Diagonal([4, 1, -2])
    @test anharmonicity_disorder(basis; pattern = [2, -1]) == Diagonal([-2, 0, 1])
    @test diagonal_operator(basis, f -> sum(f)) == [2, 2, 2]

    @test operator(basis, f -> ([copy(f), copy(f)], [0.25, 0.75])) == I
    @test operator(basis, f -> (f, 1im); element_type = ComplexF64) == 1im * I(3)
    @test iszero(operator(basis, f -> (Int[], 0.0)))

    grid = Basis_constant_N(4, 1)
    square = [0 1 1 0; 1 0 0 1; 1 0 0 1; 0 1 1 0]
    @test hopping(grid; dims = (2, 2)) == square
    @test hopping(grid; dims = (2, 2), periodic = 1) == [0 2 1 0; 2 0 0 1; 1 0 0 2; 0 1 2 0]
    @test hopping(grid; dims = (2, 2), periodic = 2) == 2square
    @test_throws ArgumentError hopping(grid; dims = (3, 2))
    @test_throws ArgumentError hopping(grid; dims = (2, 2), periodic = 3)

    hardcore = Basis_local_max_N(2, 1)
    @test iszero(hopping(hardcore) * product_state(hardcore, [1, 1]))
end
