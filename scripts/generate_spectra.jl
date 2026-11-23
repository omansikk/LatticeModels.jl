using LinearAlgebra, SparseArrays, DelimitedFiles, LatticeModels, Base.Threads

BLAS.set_num_threads(1)

pop1(fock, site) = (fock, fock[site] == 1 ? 1 : 0)


function tls_hops(basis, fock, pattern)
    focks_out::Vector{Vector{Int64}} = []
    elements::Vector{Float64} = []
    copy_fock = copy(fock)
    for l in 2:basis.L
        if fock[1] > 0 && fock[l] == 0
            copy_fock .= copy(fock)
            copy_fock[1] -= 1
            copy_fock[l] += 1
            if find_index(basis, fock) <= length(basis)
                push!(focks_out, copy(copy_fock))
                push!(elements, sqrt(fock[1]) * pattern[l - 1])
            end
        end
    end

    return focks_out, elements
end


function tls_hamiltonian(basis, ns, as, n, pars, A)
    d = length(basis)
    H0 = zeros(d, d)
    H0 .+= pars[1][1] .* ns[1] .- (0.5 * pars[1][2]) * (ns[1] .* (ns[1] .- I(d)))
    ntls = length(pars) - 1
    pattern = zeros(ntls)
    for i in 1:ntls
        H0 .+= pars[i + 1][1] .* ns[i + 1]
        pattern[i] = pars[i + 1][2]
    end

    H0 .+= operator(basis, fock -> tls_hops(basis, fock, pattern), add_adjoint = true)
    H0 .+= 0.5 * A .* (as[1] .+ as[1]')

    return H0
end


function optimise_pulse(basis, H, as, n, P1, state, pars, pulse_time)
    nfreq = 30
    freq = pars[1][1] .+ collect(range(-0.1pi, 0.1pi, nfreq))

    state1 = zeros(ComplexF64, length(basis))
    vals = zeros(nfreq)
    for i in 1:nfreq
        state1 .= exp(-1.0im * pulse_time .* Matrix(H .- freq[i] .* n)) * state
        vals[i] = real(state1' * P1 * state1)
    end

    freq_max_val = freq[argmax(vals)]
    freq .= collect(range(-0.02pi, 0.02pi, nfreq)) .+ freq_max_val
    for i in 1:nfreq
        state1 .= exp(-1.0im * pulse_time .* Matrix(H .- freq[i] .* n)) * state
        vals[i] = real(state1' * P1 * state1)
    end

    return H .- freq[argmax(vals)] .* n
end



function generate_spectra(ntls::Integer, realisations::Integer, name::AbstractString; nfreq::Integer = 201, ntime::Integer = 101, tmax::Real = 5e2)
    v::Vector{Basis} = [Basis_local_max_N(1, 2), Basis_local_max_N(ntls, 1)]
    basis = Basis_composite(v); d = length(basis)
    ns = numbers(basis)
    n = sum(ns)
    as = annihilations(basis)
    pulse_time = 80.; A = pi / 30.
    P1 = operator(basis, fock -> pop1(fock, 1))
    nfreq > 1 && ntime > 1 && tmax > 0 || throw(ArgumentError("invalid spectrum grid"))
    time = collect(range(0., tmax, ntime)); dt = time[2] - time[1]
    pop = zeros(ntime, nfreq)
    state = product_state(basis, zeros(Int64, 1 + ntls))
    rho_init = state * state'
    labels = zeros(realisations, (ntls + 1) * 4)
    path = name
    mkpath(path)
    H0 = spzeros(d, d)
    H1 = spzeros(d, d)
    H2 = spzeros(d, d)
    for k in 1:realisations
        pars = [[7. * 2pi, 0.18 * 2pi, 3e3, 1e3]]
        for i in 1:ntls
            push!(pars, [pars[1][1] - (0.25 - 0.5 * rand()) * 2pi, (0.005 + 0.045 * rand()) * 2pi, (0.1 + 9.9 * rand()) * 1e3, (0.1 + 9.9 * rand()) * 1e3])
        end
        
        labels[k, :] .= reduce(vcat, pars)
        freq = pars[1][1] .+ collect(range(-0.4pi, 0.4pi, nfreq))
        H0 .= tls_hamiltonian(basis, ns, as, n, pars, pi / pulse_time)
        H1 .= tls_hamiltonian(basis, ns, as, n, pars, A)
        H2 .= optimise_pulse(basis, H0, as, n, P1, state, pars, pulse_time)
        jumps = vcat([as[i] / sqrt(pars[i][3]) for i in eachindex(as) if pars[i][3] > 0], [sqrt(2 / pars[i][4]) * ns[i] for i in eachindex(ns) if pars[i][4] > 0])
        readout_generator = liouvillian(H2, jumps; representation = :dense)
        readout_observable = exp(pulse_time * readout_generator') * vec(Matrix(P1))
        observe = (rho, t) -> real(dot(readout_observable, vec(rho)))
        @threads for i in 1:nfreq
            result = run_lindblad(basis, H1 - freq[i] * n, rho_init, [observe], tmax;
                                  jump_operators = jumps, representation = :dense, method = :dense,
                                  saveat = time, dt = dt)
                
            pop[:, i] .= result.outputs[1]
        end
        
        open(joinpath(path, "population_" * string(k)); write = true) do f
            writedlm(f, reduce(vcat, pars)')
            writedlm(f, pop)
        end
    
    end

    writedlm(joinpath(path, "labels"), labels)
    writedlm(joinpath(path, "times"), time)
    writedlm(joinpath(path, "freq"), range(-0.4pi, 0.4pi; length = nfreq))
end

if abspath(PROGRAM_FILE) == @__FILE__
    @time generate_spectra(1, 10000, "TLS_data")
end
