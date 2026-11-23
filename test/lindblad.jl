using Random

@testset "Coherent solver and shared propagation" begin
    basis = Basis_constant_N(2, 1)
    H = ComplexF64[0 -im; im 0]
    psi = ComplexF64[1, 0]
    times = [0.0, 0.17, 0.6, 1.0]
    outputs = [p -> p[2], (p, t) -> norm(p), (p, t) -> t]
    for method in (:dense, :krylov, :auto)
        result = run_unitary(basis, H, psi, outputs, 1;
            saveat = times, dt = 0.3, method, store_states = true)
        @test result.outputs[1] ≈ sin.(times) atol = 1e-12
        @test result.outputs[2] ≈ ones(4) atol = 1e-12
        @test result.outputs[3] == times
        @test result.final_state ≈ [cos(1), sin(1)] atol = 1e-12
        @test length(result.states) == 4
        @test result.states[end] ≈ result.final_state
        @test result.states[end] !== result.final_state
        result.states[1][1] = 42
        @test psi == [1, 0]
        @test result.final_state[1] ≈ cos(1)
        ensemble = run_trajectories(basis, H, psi, outputs, 1; saveat = times, dt = 0.3, method, trajectories = 3)
        @test all(ensemble.outputs[k] ≈ result.outputs[k] for k in eachindex(outputs))
    end
    
    dynamic = run_unitary(basis, t -> t * H, psi, [p -> p[2]], 1; saves = 2, dt = 0.3)
    @test dynamic.outputs[1] ≈ sin.([0.0, 0.125, 0.5]) atol = 1e-12
    @test dynamic.states === nothing
    @test run_unitary(basis, H, psi, [], 0; steps = 2).final_state == psi
    @test_throws ArgumentError run_unitary(basis, H, psi, [], 1; dt = 0.1, steps = 2)
    @test_throws DimensionMismatch run_unitary(basis, H, [1], [], 1)
end

@testset "Explicit and matrix-free Liouvillians" begin
    rng = Xoshiro(1)
    M = randn(rng, ComplexF64, 3, 3)
    H = (M + M') / 2
    jumps = [randn(rng, ComplexF64, 3, 3) / 3 for k in 1:2]
    rhs = rho -> begin
        out = -im * (H * rho - rho * H)
        for C in jumps
            D = C' * C
            out += C * rho * C' - (D * rho + rho * D) / 2
        end
        
        out
    end
    
    reference = hcat([vec(rhs(reshape(Matrix{ComplexF64}(I, 9, 9)[:, j], 3, 3))) for j in 1:9]...)
    x = randn(rng, ComplexF64, 9)
    for source in (H, sparse(H)), representation in (:dense, :sparse, :matrixfree)
        L = liouvillian(source, jumps; representation)
        @test size(L) == (9, 9)
        @test size(L, 1) == 9
        @test eltype(L) == ComplexF64
        @test L * x ≈ reference * x atol = 1e-12
        y = zeros(ComplexF64, 9)
        @test mul!(y, L, x) === y
        @test y ≈ reference * x atol = 1e-12
        old = copy(x)
        @test mul!(y, L, x, 2im, 0.3) ≈ 2im * reference * x + 0.3 * reference * x atol = 1e-12
        @test x == old
        if representation == :matrixfree
            @test Matrix(L) ≈ reference atol = 1e-12
            @test sparse(L) ≈ reference atol = 1e-12
            alias = copy(x)
            mul!(alias, L, alias)
            @test alias ≈ reference * x atol = 1e-12
            alias = copy(x)
            mul!(alias, L, alias, 0.7im, 0.3)
            @test alias ≈ 0.7im * reference * x + 0.3 * x atol = 1e-12
            fill!(y, NaN)
            mul!(y, L, x, 1, 0)
            @test y ≈ reference * x atol = 1e-12
            copyto!(y, x)
            mul!(y, L, x, 0, 0.3)
            @test y ≈ 0.3 * x atol = 1e-12
            independent = copy(L)
            @test independent.workspace !== L.workspace
            @test independent * x ≈ L * x
            @test size(L, 3) == 1
            @test_throws DimensionMismatch mul!(y, L, x[1:8])
            @test_throws ArgumentError size(L, 0)
        else
            @test issparse(L) == (representation == :sparse)
        end
    end
    
    L = liouvillian(H, jumps)
    @test vec(Matrix{ComplexF64}(I, 3, 3))' * L ≈ zeros(1, 9) atol = 1e-12
    # Sparse Hamiltonians and jump matrices also support fused alpha/beta updates.
    sparse_jumps = sparse.(jumps)
    action = liouvillian(sparse(H), sparse_jumps; representation = :matrixfree)
    y = copy(x)
    mul!(y, action, x, 0.7im, 0.3)
    @test y ≈ 0.7im * reference * x + 0.3 * x atol = 1e-12
    @test liouvillian(H; representation = :dense) ≈ liouvillian(H, (); representation = :dense)
    @test_throws ArgumentError liouvillian(H; representation = :invalid)
    @test_throws ArgumentError liouvillian([0.0 1; 0 0])
    @test_throws DimensionMismatch liouvillian(H, [zeros(2, 2)])
end

@testset "Density evolution and physical invariants" begin
    basis = Basis_local_max_N(1, 1)
    a = Matrix(annihilation(basis, 1))
    H = ComplexF64[0 -0.6im; 0.6im 0.2]
    jumps = [sqrt(0.8) * a, sqrt(0.3) * a', sqrt(0.2) * [1.0 0; 0 -1]]
    rho0 = ComplexF64[0.7 0.1im; -0.1im 0.3]
    original = copy(rho0)
    times = [0.0, 0.17, 0.6, 1.0]
    outputs = [rho -> rho[1, 2], (rho, t) -> real(rho[2, 2]) + t, rho -> tr(rho)]
    L = liouvillian(H, jumps; representation = :dense)
    exact = [reshape(exp(t * L) * vec(rho0), 2, 2) for t in times]
    for representation in (:matrixfree, :sparse, :dense), method in (:auto, :krylov, :dense)
        representation == :matrixfree && method == :dense && continue
        result = run_lindblad(basis, H, rho0, outputs, 1; jump_operators = jumps, representation, method, saveat = times, dt = 0.4, store_states = true)
        @test result.times == times
        @test result.outputs[1] ≈ [rho[1, 2] for rho in exact] atol = 1e-11
        @test result.outputs[2] ≈ [real(rho[2, 2]) + t for (rho, t) in zip(exact, times)] atol = 1e-11
        @test result.final_state ≈ exact[end] atol = 1e-11
        @test rho0 == original
        for (rho, expected) in zip(result.states, exact)
            @test rho ≈ expected atol = 1e-11
            @test tr(rho) ≈ 1 atol = 1e-11
            @test rho ≈ rho' atol = 1e-11
            @test eigmin(Hermitian(rho)) >= -1e-11
        end
        
        result.states[1][1, 1] = 42
        @test result.final_state ≈ exact[end] atol = 1e-11
        @test rho0 == original
    end

    mixed = Matrix{ComplexF64}(I, 2, 2) / 2
    relaxed = run_lindblad(basis, zeros(2, 2), mixed, [rho -> real(rho[2, 2])], 2; jump_operators = [a], saves = 4, dt = 0.5, store_states = true)
    @test relaxed.outputs[1] ≈ 0.5exp.(-relaxed.times) atol = 1e-11
    @test norm(relaxed.final_state) > norm(mixed)
    @test relaxed.states[1] == mixed
    @test relaxed.final_state[1, 1] ≈ 1 - 0.5exp(-2) atol = 1e-11
    coherent = run_lindblad(basis, H, rho0, [], 1; saves = 2)
    U = exp(-im * H)
    @test coherent.final_state ≈ U * rho0 * U' atol = 1e-11
    @test coherent.states === nothing
    pure = ComplexF64[1, im] / sqrt(2)
    psi = run_unitary(basis, H, pure, [], 1; saves = 2).final_state
    density = run_lindblad(basis, H, pure * pure', [], 1; saves = 2).final_state
    @test density ≈ psi * psi' atol = 1e-11
    mc = run_trajectories(basis, H, pure, [psi -> abs2(psi[2])], 1; jump_operators = jumps, saves = 2, dt = 0.3, trajectories = 1000, rng = Xoshiro(1))
    master = run_lindblad(basis, H, pure * pure', [rho -> real(rho[2, 2])], 1; jump_operators = jumps, saves = 2, dt = 0.3)
    @test all(abs.(mc.outputs[1] - master.outputs[1]) .<= 5mc.standard_errors[1] .+ 0.01)
end

@testset "Time-dependent Lindblad evolution" begin
    basis = Basis_local_max_N(1, 1)
    a = Matrix(annihilation(basis, 1))
    excited = ComplexF64[0 0; 0 1]
    times = [0.0, 0.17, 0.6, 1.0]
    for representation in (:matrixfree, :sparse, :dense)
        result = run_lindblad(basis, zeros(2, 2), excited, [(rho, t) -> real(rho[2, 2]) + t], 1;
            jump_operators = t -> [sqrt(t) * a], representation, saveat = times, dt = 0.3)
        @test result.outputs[1] ≈ exp.(-times.^2 / 2) + times atol = 1e-11
    end
    
    X = ComplexF64[0 1; 1 0]
    Z = ComplexF64[1 0; 0 -1]
    H = t -> X + t^2 * Z
    
    # Static products and dynamic Hamiltonian; then both fields time dependent.
    for jumps in ([sqrt(0.3) * a], t -> [sqrt(0.3 + t) * a, sqrt(0.2t) * a'])
        dense = run_lindblad(basis, H, excited, [rho -> rho[1, 2]], 1; jump_operators = jumps, representation = :dense, saveat = times, dt = 0.2)
        free = run_lindblad(basis, H, excited, [rho -> rho[1, 2]], 1; jump_operators = jumps, saveat = times, dt = 0.2)
        @test free.outputs[1] ≈ dense.outputs[1] atol = 1e-8
        @test free.final_state ≈ dense.final_state atol = 1e-8
    end
end

@testset "Lindblad validation and pulse readout" begin
    basis = Basis_local_max_N(1, 1)
    a = Matrix(annihilation(basis, 1))
    H = ComplexF64[0 -im; im 0]
    rho0 = ComplexF64[1 0; 0 0]
    run(rho = rho0; kwargs...) = run_lindblad(basis, H, rho, [], 1; kwargs...)
    @test_throws ArgumentError run(representation = :invalid)
    @test_throws ArgumentError run(method = :dense)
    @test_throws ArgumentError run([2.0 0; 0 -1])
    @test_throws ArgumentError run([0.5 0.2; 0 0.5])
    @test_throws ArgumentError run([0.3 0; 0 0.3])
    @test_throws ArgumentError run([NaN 0; 0 1])
    @test_throws DimensionMismatch run(ones(3, 3))
    @test_throws DimensionMismatch run(jump_operators = [zeros(3, 3)])
    @test_throws ArgumentError run(dt = -1)
    @test_throws MethodError run(outputs = [])
    @test run_lindblad(basis, H, rho0, [], 0; saves = 1).final_state == rho0
    @test run(check_physical = false).final_state ≈ run().final_state

    # Fixed readout pulse moved onto the observable, as in generate_spectra.jl.
    jumps = [sqrt(0.3) * a]
    P = ComplexF64[0 0; 0 1]
    Lreadout = liouvillian(0.4H, jumps; representation = :dense)
    effective_observable = exp(0.7Lreadout') * vec(P)
    result = run_lindblad(basis, H, rho0, [rho -> real(dot(effective_observable, vec(rho)))], 1; jump_operators = jumps, saves = 3, store_states = true)
    for (value, rho) in zip(result.outputs[1], result.states)
        readout = run_lindblad(basis, 0.4H, rho, [r -> real(tr(P * r))], 0.7; jump_operators = jumps, saves = 1)
        @test value ≈ readout.outputs[1][end] atol = 1e-10
    end
end
