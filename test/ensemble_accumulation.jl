using Random

@testset "Mixed-type ensemble statistics" begin
    rng = Xoshiro(1)
    samples = [Vector[randn(rng, 17), randn(rng, ComplexF64, 17)] for i in 1:7]
    means = deepcopy(first(samples))
    deviations = [zeros(17), zeros(17)]
    for count in 2:length(samples)
        LatticeModels._accumulate_outputs!(means, deviations, samples[count], count)
    end
    
    for k in eachindex(means)
        expected = sum(sample[k] for sample in samples) / length(samples)
        squared_errors = sum(abs2.(sample[k] - expected) for sample in samples)
        @test means[k] ≈ expected atol = 1e-12
        @test deviations[k] ≈ squared_errors atol = 1e-12
    end
    
    @test eltype(means[1]) == Float64
    @test eltype(means[2]) == ComplexF64
    @test_throws DimensionMismatch LatticeModels._accumulate_outputs!(means, deviations[1:1], first(samples), 2)
    @test_throws DimensionMismatch LatticeModels._accumulate_output!(zeros(3), zeros(2), ones(3), 2)

    basis = Basis_local_max_N(1, 1)
    H = ComplexF64[0 0.3im; -0.3im 0.1]
    jumps = [sqrt(0.8) * Matrix(annihilation(basis, 1))]
    state = ComplexF64[1, im] / sqrt(2)
    outputs = [psi -> abs2(psi[2]), psi -> conj(psi[1]) * psi[2]]
    times = [0.0, 0.2, 0.4]
    count = 12
    seed = 10
    result = run_trajectories(basis, H, state, outputs, 0.4; jump_operators = jumps, saveat = times, dt = 0.1, trajectories = count, rng = Xoshiro(seed))
    options = (; dt = 0.1, method = :auto, krylov_dimension = 0, atol = 1e-10, rtol = 1e-8, jump_tol = 1e-10, max_steps = 10^7)
    callbacks = LatticeModels._output_callbacks(outputs, state)
    seeds = rand(Xoshiro(seed), UInt64, count)
    runs = [LatticeModels._trajectory(H, jumps, state, callbacks, times, options, Xoshiro(s)) for s in seeds]
    for k in eachindex(outputs)
        expected = sum(run[k] for run in runs) / count
        standard_error = sqrt.(sum(abs2.(run[k] - expected) for run in runs) / (count * (count - 1)))
        @test result.outputs[k] ≈ expected atol = 1e-12
        @test result.standard_errors[k] ≈ standard_error atol = 1e-12
    end
end


@testset "Fused effective-generator construction" begin
    rng = Xoshiro(1)
    A = randn(rng, ComplexF64, 4, 4)
    H0 = (A + A') / 2
    C0 = [randn(rng, ComplexF64, 4, 4) for i in 1:3]
    for H in (H0, sparse(H0)), jumps in (C0, sparse.(C0))
        original = copy(H)
        expected = -im * H
        for C in jumps
            expected = expected - 0.5 * (C' * C)
        end
        
        @test LatticeModels._effective_generator(H, jumps) ≈ expected atol = 1e-12
        @test H == original
    end
end
