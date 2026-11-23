using Test
using LatticeModels
using LinearAlgebra
using SparseArrays

@testset "LatticeModels" begin
    include("basis.jl")
    include("operators.jl")
    include("observables.jl")
    include("transforms.jl")
    include("dynamics.jl")
    include("lindblad.jl")
end
