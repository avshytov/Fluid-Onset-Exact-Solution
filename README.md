# Fluid-Onset-Exact-Solution

The code in this project implements a computational model of e-fluid confined to a half-plane.
The quantum Boltzmann equation, with a simplistic choice of the momentum-conserving collision
integral, admits an exact solution via the Wiener-Hopf method, as outlined in the notes. 
The solution is applicable at all relevant scales: shorter than the mean-free path, much larger
than the mean-free path, and of the order of the mean-free path. Thereby, one can employ 
the model implemented here to study the crossover between ballistic transport and fluidity, 
as well as the ohmic regime. 

This work was supported by the EPSRC grant EP/T001194/1
