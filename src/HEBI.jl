module HEBI

using Base.Iterators
using Distributions, Distances
using FastClosures
using LinearAlgebra
using LyceumAI

using LyceumBase
using LyceumBase: _rollout
using LyceumBase.Tools

using LyceumMuJoCo
using LyceumMuJoCo: reset_nofwd!
#using LyceumMuJoCoViz
#using UniversalLogger

using MuJoCo
using MuJoCo.MJCore: libmujoco, mjtNum, mjtByte, mjModel, mjData, Model, Data, MJVec

using Parameters
using Random
using Rotations
using Shapes
using StaticArrays
using Statistics
using UnsafeArrays


const AbsMat = AbstractMatrix
const AbsVec = AbstractVector
#using Printf
#using CSV

#using Optim
#using LineSearches

#using BlackBoxOptim
#using Distributed
#using SharedArrays

include("hebipickup.jl")
include("util.jl")
#include("vizmodes.jl")
include("jacobianmethods.jl")

end
