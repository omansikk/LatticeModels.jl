using Random, Distributed


@testset "Deterministic time evolution" begin
    basis = Basis_constant_N(2, 1)
    initial = product_state(basis, [1, 0])
    original = copy(initial)
    H = hopping(basis)
    nleft = number(basis, 1)
    outputs = [psi -> real(dot(psi, nleft, psi)), (psi, t) -> norm(psi), (psi, t) -> t]
    times = collect(range(0.0, 1.0; length = 6))
    for hamiltonian in (Matrix(H), H), method in (:dense, :krylov, :auto)
        result = run_trajectories(basis, hamiltonian, initial, outputs, 1; saveat = times, steps = 10, method, krylov_dimension = 2)
        @test result.times == times
        @test result.outputs[1] ≈ cos.(times).^2 atol = 1e-12
        @test result.outputs[2] ≈ ones(6) atol = 1e-12
        @test result.outputs[3] == times
        @test all(isnan, result.standard_errors[1])
        @test initial == original
    end
    
    repeated = run_trajectories(basis, H, initial, outputs, 1; saves = 5, trajectories = 3)
    @test repeated.outputs[1] ≈ cos.(times).^2 atol = 1e-12
    @test repeated.standard_errors[1] ≈ zeros(6) atol = 1e-14

    Y = ComplexF64[0 -im; im 0]
    for method in (:dense, :krylov)
        result = run_trajectories(basis, Y, initial, [psi -> psi[2]], 1; saveat = times, method, krylov_dimension = 2)
        @test result.outputs[1] ≈ sin.(times) atol = 1e-12
        eigenstate = ComplexF64[1, im] / sqrt(2)
        stationary = run_trajectories(basis, sparse(Y), eigenstate, [psi -> psi[1]], 1; saveat = times, method, krylov_dimension = 2)
        @test stationary.outputs[1] ≈ exp.(-im .* times) / sqrt(2) atol = 1e-12
    end
end


@testset "Truncated Arnoldi convergence" begin
    basis = Basis_constant_N(5, 1)
    H = ComplexF64[0 im 0 0 0; -im 0 0.7 0 0; 0 0.7 0 1 0; 0 0 1 0 0.4; 0 0 0 0.4 0]
    initial = product_state(basis, [1, 0, 0, 0, 0])
    callbacks = [psi -> psi[k] for k in 1:5]
    jumps = [0.1 * Matrix{ComplexF64}(I, 5, 5)]
    dense = run_trajectories(basis, H, initial, callbacks, 0.6; jump_operators = jumps, saves = 3, dt = 0.3, method = :dense, rng = Xoshiro(1))
    krylov = run_trajectories(basis, sparse(H), initial, callbacks, 0.6; jump_operators = jumps, saves = 3, dt = 0.3, 
                              method = :krylov, krylov_dimension = 3, atol = 1e-12, rtol = 1e-10, rng = Xoshiro(1))
                              
    @test all(isapprox(dense.outputs[k], krylov.outputs[k]; atol = 1e-7) for k in 1:5)
end


@testset "Time dependence and exact observation times" begin
    basis = Basis_constant_N(2, 1)
    initial = product_state(basis, [1, 0])
    X = Matrix(hopping(basis))
    times = [0.0, 0.13, 0.51, 1.0]
    H = t -> t * X
    outputs = [(psi, t) -> abs2(psi[1]), (psi, t) -> t + abs2(psi[1]), (psi, t) -> real(dot(psi, H(t), psi))]
    for method in (:dense, :krylov)
        result = run_trajectories(basis, H, initial, outputs, 1; saveat = times[2:end-1], dt = 0.3, method)
        expected = cos.(times.^2 / 2).^2
        @test result.times == times
        @test result.outputs[1] ≈ expected atol = 1e-11
        @test result.outputs[2] ≈ times + expected atol = 1e-11
        @test result.outputs[3] ≈ zeros(4) atol = 1e-11
    end
    
    # A noncommuting driven Hamiltonian should converge as the maximum step shrinks.
    Z = [1. 0.; 0. -1.]
    driven = t -> X + t^2 * Z
    amplitudes = [psi -> psi[1], psi -> psi[2]]
    reference = run_trajectories(basis, driven, initial, amplitudes, 1; saveat = [1.0], dt = 0.002, atol = 1e-8, rtol = 1e-8)
    errors = Float64[]
    for dt in (0.2, 0.1)
        result = run_trajectories(basis, driven, initial, amplitudes, 1; saveat = [1.0], dt, atol = 1.0, rtol = 1.0)
        push!(errors, norm([result.outputs[k][end] - reference.outputs[k][end] for k in 1:2]))
    end
    
    @test errors[2] < errors[1] / 3
end


@testset "Quantum jumps, heating and dephasing" begin
    basis = Basis_local_max_N(1, 1)
    a = Matrix(annihilation(basis, 1))
    n = Matrix(number(basis, 1))
    H = zeros(ComplexF64, 2, 2)
    ground = product_state(basis, [0])
    excited = product_state(basis, [1])
    occupation = [(psi, t) -> real(dot(psi, n, psi)), psi -> norm(psi)]
    times = [0.0, 0.4, 1.0]
    for (state, jumps, expected) in (
        (excited, [a], exp.(-times)),
        (ground, [sqrt(0.7) * a'], 1 .- exp.(-0.7times)),
        (ground, [sqrt(1.3) * a, sqrt(0.7) * a'], 0.35 .* (1 .- exp.(-2times))),
    )
    
        result = run_trajectories(basis, H, state, occupation, 1; jump_operators = jumps, saveat = times, dt = 1, trajectories = 1200, rng = Xoshiro(1))
        @test all(abs.(result.outputs[1] - expected) .<= 5result.standard_errors[1] .+ 0.01)
        @test result.outputs[2] ≈ ones(3) atol = 1e-12
        @test all(isfinite, result.standard_errors[1])
    end
    
    plus = (ground + excited) / sqrt(2)
    Z = [1.0 0; 0 -1]
    dephased = run_trajectories(basis, H, plus, [psi -> real(conj(psi[1]) * psi[2])], 1; jump_operators = [sqrt(0.4) * Z], saveat = times, dt = 1, trajectories = 1200, rng = Xoshiro(1))
    @test all(abs.(dephased.outputs[1] - 0.5exp.(-0.8times)) .<= 5dephased.standard_errors[1] .+ 0.01)

    # Compare both propagation methods for exactly the same random streams.
    for method in (:dense, :krylov)
        result = run_trajectories(basis, H, excited, occupation, 1; jump_operators = [a], saveat = times, dt = 1, trajectories = 20, method, rng = Xoshiro(1))
        repeat = run_trajectories(basis, H, excited, occupation, 1; jump_operators = [a], saveat = times, dt = 1, trajectories = 20, method, rng = Xoshiro(1))
        @test result.outputs == repeat.outputs
        @test result.standard_errors == repeat.standard_errors
    end

    # Identity-strength flips form a Poisson process. Reproduce the exact stream,
    # including channel-selection draws, to check multiple jumps in one interval.
    parent = Xoshiro(1)
    local_rng = Xoshiro(rand(parent, UInt64))
    event_time = -log(rand(local_rng)) / 20
    flips = 0
    while event_time <= 1
        flips += 1
        rand(local_rng) # channel selection
        event_time += -log(rand(local_rng)) / 20
    end
    
    @test flips > 1
    X = [0.0 1; 1 0]
    for method in (:dense, :krylov)
        result = run_trajectories(basis, H, ground, occupation, 1; jump_operators = [sqrt(20) * X], saveat = [1.0], dt = 1, method, rng = Xoshiro(1))
        @test result.outputs[1][end] ≈ flips % 2 atol = 1e-12
        @test result.outputs[2] ≈ ones(2) atol = 1e-12
    end

    # Time-dependent jump amplitude: survival exp(-integral_0^t s ds).
    result = run_trajectories(basis, H, excited, occupation, 1; jump_operators = t -> [sqrt(t) * a], 
                              saveat = times, dt = 0.4, trajectories = 800, rng = Xoshiro(1))
    @test all(abs.(result.outputs[1] - exp.(-times.^2 / 2)) .<= 5result.standard_errors[1] .+ 0.01)

    # A dark state must remain valid, including when all channels have zero weight.
    dark = run_trajectories(basis, H, ground, occupation, 1; jump_operators = [a, zeros(2, 2)], saveat = times)
    @test dark.outputs[1] == zeros(3)
    @test dark.outputs[2] ≈ ones(3)
end


@testset "Independent Lindblad reference" begin
    basis = Basis_local_max_N(1, 1)
    a = Matrix(annihilation(basis, 1))
    H = ComplexF64[0 -0.6im; 0.6im 0.2]
    jumps = [sqrt(0.8) * a, sqrt(0.3) * a', sqrt(0.2) * [1.0 0; 0 -1]]
    initial = ComplexF64[1, im] / sqrt(2)
    rho = initial * initial'
    eye = Matrix{ComplexF64}(I, 2, 2)
    liouvillian = -im * (kron(eye, H) - kron(transpose(H), eye))
    for C in jumps
        D = C' * C
        liouvillian += kron(conj(C), C) - (kron(eye, D) + kron(transpose(D), eye)) / 2
    end
    
    times = [0.0, 0.3, 0.8]
    observables = [ComplexF64[0 0; 0 1], ComplexF64[0 1; 0 0]]
    callbacks = [psi -> dot(psi, O, psi) for O in observables]
    result = run_trajectories(basis, H, initial, callbacks, 0.8; jump_operators = jumps, saveat = times, dt = 0.2, trajectories = 1600, rng = Xoshiro(1))
    for (i, t) in enumerate(times), k in eachindex(observables)
        density = reshape(exp(t * liouvillian) * vec(rho), 2, 2)
        exact = tr(observables[k] * density)
        @test abs(result.outputs[k][i] - exact) <= 5result.standard_errors[k][i] + 0.01
    end
    
    # Full two-dimensional Arnoldi must reproduce the non-Hermitian dense action.
    dense = run_trajectories(basis, H, initial, callbacks, 0.8; jump_operators = jumps, saveat = times, dt = 0.2, trajectories = 30, rng = Xoshiro1))
    krylov = run_trajectories(basis, sparse(H), initial, callbacks, 0.8; jump_operators = jumps, saveat = times, dt = 0.2, method = :krylov, trajectories = 30, rng = Xoshiro(1))
    @test all(dense.outputs[k] ≈ krylov.outputs[k] for k in eachindex(callbacks))
    @test H == ComplexF64[0 -0.6im; 0.6im 0.2]
    @test jumps[1] == sqrt(0.8) * a
end


@testset "Validation, realizations and distributed execution" begin
    basis = Basis_constant_N(2, 1)
    H = Matrix(hopping(basis))
    initial = product_state(basis, [1, 0])
    outputs = [psi -> abs2(psi[1])]
    run(; kwargs...) = run_trajectories(basis, H, initial, outputs, 1; kwargs...)
    @test_throws DimensionMismatch run(jump_operators = [zeros(3, 3)])
    @test_throws ArgumentError run(saveat = [0.5, 0.2])
    @test_throws ArgumentError run(saveat = [-1.0])
    @test_throws ArgumentError run(dt = 0)
    @test_throws ArgumentError run(dt = 0.1, steps = 10)
    @test_throws ArgumentError run(trajectories = 0)
    @test_throws ArgumentError run(method = :invalid)
    @test_throws ArgumentError run(method = :krylov, krylov_dimension = 1)
    @test_throws ArgumentError run(rtol = NaN)
    @test_throws ArgumentError run_trajectories(basis, [0.0 1; 0 0], initial, outputs, 1)
    @test_throws ArgumentError run_trajectories(basis, H, zeros(2), outputs, 1)
    @test run_trajectories(basis, H, initial, outputs, 0).times == [0.0]
    @test run(saveat = [0.4], dt = 0.7).times == [0.0, 0.4, 1.0]
    @test run_trajectories(basis, H, initial, [], 1; saves = 2).outputs == []

    calls = Ref(0)
    factory = rng -> begin
        calls[] += 1
        (; hamiltonian = rand(rng) * H, jump_operators = ())
    end
    
    result = run(realization = factory, trajectories = 4, saves = 2, rng = Xoshiro(1))
    @test calls[] == 4
    @test all(isfinite, result.outputs[1])

    # A shared factory tests disorder and jumps with the same per-index streams.
    factory2 = rng -> (; hamiltonian = rand(rng) * H, jump_operators = [sqrt(0.7) * [0.0 1; 1 0]])
    serial = run(realization = factory2, trajectories = 8, saves = 3, dt = 0.3, rng = Xoshiro(1))
    added = addprocs(2; exeflags = `--project=$(dirname(Base.active_project()))`)
    try
        for worker in added
            remotecall_wait(Base.eval, worker, Main, :(using LatticeModels))
        end
        
        distributed = run(realization = factory2, trajectories = 8, saves = 3, dt = 0.3, parallel = :distributed, rng = Xoshiro(1))
        @test distributed.times == serial.times
        @test distributed.outputs == serial.outputs
        @test distributed.standard_errors == serial.standard_errors
    finally
        rmprocs(added)
    end
end
