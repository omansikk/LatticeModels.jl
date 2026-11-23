# LatticeModels
This is a collection of most of the code used in the following papers:

[1](https://link.aps.org/doi/10.1103/PhysRevB.103.L220202) O. Mansikkamäki, S. Laine, and M. Silveri, Phases of the disordered Bose-Hubbard model with attractive interactions, Phys. Rev. B 103, L220202 (2021).

[2](https://link.aps.org/doi/10.1103/PRXQuantum.3.040314) O. Mansikkamäki et al., Beyond Hard-Core Bosons in Transmon Arrays, PRX Quantum 3, 040314 (2022).

[3](https://link.aps.org/doi/10.1103/PhysRevResearch.5.023121) O. Busel et al., Dissipation and Dephasing of Interacting Photons in Transmon Arrays, Phys. Rev. Res. 5, 023121 (2023)

See Examples.ipynb to see how it works, and how to use it to reproduce some of the results of the above papers. Note that the package was originally quite quickly cobbled together from three separate application specific codes, and was buggy. Those bugs should be fixed now, and some new bugs no doubt were added in the process.

The old and messy `run_trajectories` Trajectories.jl has been replaced with a new set of functions: 
`run_trajectories` still runs quantum trajectories for decoherent evolution. 
`run_unitary` now handles unitary evolution. 
`run_lindblad` has been added to run decoherent evolution with the full Lindblad equation.

All of them should be capable of working with time-dependent Hamiltonians and output functions.

A script for generating simulated spectroscopic data used in https://arxiv.org/abs/2404.14039 is available in the folder `scripts`.
