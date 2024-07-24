"""
Kuramoto Sivashinsky Equation 1D
"""

using Plots
using OrdinaryDiffEq
using JLD2
using BlockArrays
using LinearAlgebra
using SparseArrays

function ks_fd_operators(n, dx)
    ∂x = (diagm(1=>ones(n-1)) + diagm(-1=>-1*ones(n-1)))
    ∂x[1,end] = -1
    ∂x[end,1] = 1
    ∂x ./= (2*dx)
    ∂x = sparse(∂x)
    ∂x2 = diagm(0=>-2*ones(n)) + diagm(-1=>ones(n-1)) + diagm(1=>ones(n-1))
    ∂x2[1,end] = 1
    ∂x2[end,1] = 1
    ∂x2 ./= (dx)^2
    ∂x2 = sparse(∂x2)
    ∂x4 = sparse(∂x2*∂x2)
    return ∂x, ∂x2, ∂x4
end


function ks_oop(u,p,t)
    -∂x4*u - ∂x2*u - u.*(∂x*u)
end


#Define variables
n = 1024
L = 560
dt = 0.05
N_t = 2000
dx = L/n
t_start = 200 #length of initial transient to discard
t_end = t_start + N_t*dt

∂x, ∂x2, ∂x4 = ks_fd_operators(n, dx)

Nensembles = 10

for i=1:Nensembles
    println("Ensemble ID:", i)
    u0 = 0.01*(rand(Float32,n) .- 0.5) #initial condition
    prob = ODEProblem(ks_oop, Float32.(u0), (0.,t_end))
    println("Solving..")
    sol = Array(solve(prob, Tsit5(), saveat=t_start:dt:t_end))
    fname = "ks-solutionEns$i.jld2"
    @save fname sol
    println("Done!")
end
