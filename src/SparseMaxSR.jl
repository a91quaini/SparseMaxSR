module SparseMaxSR

using JuMP
using Random, LinearAlgebra
using DataFrames, CSV
using StatsBase
using LaTeXStrings
using Suppressor           # for @suppress in tests / demos
# Bridging packages for solvers
import CPLEX, Mosek

# ──────────────────────────────────────────────────────────
# Public API

end # module