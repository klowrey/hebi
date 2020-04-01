module HEBI

using Base.Iterators
using Distances
using Distributions
using FastClosures
using LinearAlgebra
using LyceumAI
using LyceumBase
using LyceumBase: _rollout
using LyceumBase.Tools

using LyceumMuJoCo
using LyceumMuJoCo: reset_nofwd!

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

export HEBIPickup
include("hebipickup.jl")

include("util.jl")

export dls_jacctrl!, pinv_jacctrl!, transpose_jacctrl!, sdld_jacctrl!
include("jacobianmethods.jl")

end
