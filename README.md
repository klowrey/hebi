# hebi

# NOTE: Frames are noted by the following suffixes: w = world, c = chopstick, p = pose # TODO

ROS is bad.

# Setup

Background: https://docs.lyceum.ml/dev/

```julia
bash> cd hebi
bash> julia --project
julia> ]
(hebi) pkg> registry add https://github.com/Lyceum/LyceumRegistry
(hebi) pkg> instantiate
(hebi) pkg> resolve
```

# Testing

`testsysid` function currently compares two models after setting one to have
 the wrong damping values for two joints.

It correctly recovers the correct values, assuming the same parameters for the
two different motors.

```julia
julia> include("src/runhebi.jl")
julia> h = HEBIPickup()
julia> h2 = HEBIPickup()
julia> a = getsincontrols(h, 2000)
julia> result = testsysid(h, h2, a; optm=:LBFGS)
julia> result.minimizer
6-element Array{Float64,1}:
 0.4989704738104854
 0.09929717145680375
 0.010059651430875389
 0.009927101543924597
 0.5005574377220603
 0.1057254675069472
```


The `gradsysid` function is an example of LBFGS optimization with a finite differenced
graddient function that uses multi-threading. It may be currently buggy.


# SysID

```julia
julia> include("src/runhebi.jl")
julia> h = HEBIPickup()
julia> dt, pos, vel, act, eff = datafile("data/example2.csv")
julia> result = filesysid(vcat(pos, vel), h, act; optm=:LBFGS, batch=6000:10000)
julia> hebivars = gethebivars()
julia> hv = hebivars(result.minimizer)
julia> hv.x8_damping
```
