using Random

@testset "Driven evolution with fixed jump operators" begin
    basis = Basis_local_max_N(1, 1)
    a = Matrix(annihilation(basis, 1))
    H = t -> ComplexF64[0.2t 0.4im; -0.4im -0.1t]
    jumps = [sqrt(0.8) * a, sqrt(0.3) * a', sqrt(0.2) * [1.0 0; 0 -1]]
    original = deepcopy(jumps)
    initial = ComplexF64[1, im] / sqrt(2)
    times = [0.0, 0.13, 0.4]
    common = (; saveat = times, dt = 0.1, atol = 1e-12, rtol = 1e-10)

    for method in (:dense, :krylov)
        outputs = [psi -> abs2(psi[2]), psi -> conj(psi[1]) * psi[2]]
        cached = run_trajectories(basis, H, initial, outputs, 0.4; common..., jump_operators = jumps, method, trajectories = 12, rng = Xoshiro(1))
        uncached = run_trajectories(basis, H, initial, outputs, 0.4; common..., jump_operators = t -> jumps, method, trajectories = 12, rng = Xoshiro(1))
        @test all(isapprox(a, b; atol = 1e-7) for (a, b) in zip(cached.outputs, uncached.outputs))
    end

    density = initial * initial'
    for representation in (:dense, :sparse, :matrixfree)
        cached = run_lindblad(basis, H, density, [], 0.4; common..., jump_operators = jumps, representation)
        uncached = run_lindblad(basis, H, density, [], 0.4; common..., jump_operators = t -> jumps, representation)
        @test cached.final_state ≈ uncached.final_state atol = 1e-8
    end

    @test jumps == original
    coherent = run_unitary(basis, H, initial, [psi -> abs2(psi[2])], 0.4; common...)
    no_jumps = run_trajectories(basis, H, initial, [psi -> abs2(psi[2])], 0.4; common...)
    @test no_jumps.outputs[1] ≈ coherent.outputs[1] atol = 1e-8
    density_no_jumps = run_lindblad(basis, H, density, [], 0.4; common..., representation = :dense)
    @test density_no_jumps.final_state ≈ coherent.final_state * coherent.final_state' atol = 1e-8
end
