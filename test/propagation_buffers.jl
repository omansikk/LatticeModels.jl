using Random

@testset "Reusable exponential actions and rejection safety" begin
    rng = Xoshiro(1)
    dim = 8
    basis = Basis_constant_N(dim, 1)
    M = randn(rng, ComplexF64, dim, dim)
    H = (M + M') / 2
    state = normalize!(randn(rng, ComplexF64, dim))
    initial = copy(state)
    for method in (:dense, :krylov)
        model = LatticeModels._unitary_model(H, dim)
        work = LatticeModels._propagation_workspace(model, method, dim)
        output = similar(state)
        generator = -im * H
        for h in (0.13, 0.07, 0.13), cache in (false, true)
            LatticeModels._exponential_action!(output, generator, state, h, work, cache)
            @test output ≈ exp(h * generator) * state atol = 1e-12
            @test state == initial
        end
        
        eigenstate = Vector{ComplexF64}(eigen(Hermitian(H)).vectors[:, 1])
        LatticeModels._exponential_action!(output, generator, eigenstate, 0.1, work, false)
        @test output ≈ exp(0.1generator) * eigenstate atol = 1e-12
    end
    
    X, Z = ComplexF64[0 1; 1 0], ComplexF64[1 0; 0 -1]
    model = LatticeModels._unitary_model(t -> X + t^2 * Z, 2)
    work = LatticeModels._propagation_workspace(model, :dense, 0)
    input = ComplexF64[1, 0]
    candidate, h, proposed = LatticeModels._accepted_step(model, input, 0.0, 1.0, work, 1e-12, 1e-10)
    @test h < 1.0
    @test input == [1, 0]
    @test norm(candidate) ≈ 1 atol = 1e-12
    snapshot = copy(candidate)
    LatticeModels._accepted_step(model, input, 0.2, 0.1, work, 1e-12, 1e-10)
    @test input == [1, 0]
    @test snapshot !== candidate
    @test LatticeModels._next_step_size(0.1, 1e-12, 1e-12, 2e-12) == 0.1
    @test LatticeModels._next_step_size(0.1, 0.02, 0.01, 0.01) == 0.01
end

@testset "Driven generator storage and commutator buffers" begin
    basis = Basis_constant_N(3, 1)
    X = ComplexF64[0 1im 0; -1im 0 0.5; 0 0.5 0]
    Z = Diagonal([0.3, -0.2, 0.1])
    H = t -> X + t * Z
    jumps = [ComplexF64[0 0.4 0; 0 0 0.3im; 0 0 0]]
    for representation in (:dense, :sparse, :matrixfree)
        model = LatticeModels._lindblad_model(H, jumps, 3, representation)
        for t in (0.0, 0.17, 0.4)
            actual = LatticeModels._generator(model, t)
            expected = liouvillian(H(t), jumps; representation = :dense)
            if representation == :matrixfree
                @test Matrix(actual) ≈ expected atol = 1e-12
            else
                @test actual ≈ expected atol = 1e-12
            end
        end
    end
    
    for as_matrix in (identity, sparse)
        model = LatticeModels._unitary_model(t -> as_matrix(H(t)), 3)
        first = LatticeModels._generator(model, 0.1)
        second = LatticeModels._generator(model, 0.2)
        @test first === second
        @test second ≈ -im * H(0.2) atol = 1e-12
    end
    
    psi = ComplexF64[1, 0, 0]
    for callback in (t -> sparse(Matrix(t * X)), t -> t < 0.2 ? sparse(t * X) : t * X,
        t -> t < 0.2 ? t * X : sparse(t * X))
        result = run_unitary(basis, callback, psi, [], 0.4; saveat = [0.0, 0.2, 0.4], dt = 0.1)
        @test result.final_state ≈ exp(-im * 0.4^2 / 2 * X) * psi atol = 1e-11
    end
    
    diagonal = run_unitary(basis, t -> Diagonal(t .* [1.0, 2, 3]), ones(3), [], 0.4; saves = 3, dt = 0.1, method = :krylov)
    @test diagonal.final_state ≈ exp.(-im * 0.4^2 / 2 .* [1, 2, 3]) / sqrt(3) atol = 1e-11
end

@testset "Adaptive output boundaries and owned saved states" begin
    basis = Basis_local_max_N(1, 1)
    X, Z = ComplexF64[0 1; 1 0], ComplexF64[1 0; 0 -1]
    initial = ComplexF64[1, 0]
    H0, H1 = X, Z
    original0, original1 = copy(H0), copy(H1)
    calls = Ref(0)
    H = t -> begin
        calls[] += 1
        H0 + sin(3t) * H1
    end
    
    reference = run_unitary(basis, H, initial, [], 0.4; saveat = [0.4], dt = 0.0005, atol = 1e-12, rtol = 1e-10)
    callbacks = [(psi, t) -> t, psi -> abs2(psi[2])]
    for method in (:dense, :krylov)
        times = [0.0, 0.11, 0.110000000001, 0.23, 0.4]
        result = run_unitary(basis, H, initial, callbacks, 0.4; saveat = times, dt = 0.1, atol = 1e-12, rtol = 1e-10, method, store_states = true)
        @test result.times == times
        @test result.outputs[1] == times
        @test result.final_state ≈ reference.final_state atol = 2e-7
        @test all(result.states[i] !== result.states[j] for i in eachindex(times), j in eachindex(times) if i != j)
        @test result.final_state !== result.states[end]
        result.states[1][1] = 42
        @test initial == [1, 0]
        @test norm(result.final_state) ≈ 1 atol = 1e-12
    end
    
    @test H0 == original0
    @test H1 == original1
    calls[] = 0
    run_unitary(basis, H, initial, [], 0.4; saveat = [0.0, 0.11, 0.23, 0.4], dt = 0.1, atol = 1e-12, rtol = 1e-10)
    ordinary_calls = calls[]
    calls[] = 0
    run_unitary(basis, H, initial, [], 0.4; saveat = [0.0, 0.11, 0.110000000001, 0.23, 0.4], dt = 0.1, atol = 1e-12, rtol = 1e-10)
    @test calls[] <= ordinary_calls + 12
end
