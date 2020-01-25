# hebi

ROS is bad.

# Setup

Background: https://docs.lyceum.ml/dev/

```julia
julia> ]
(hebi) pkg> registry add https://github.com/Lyceum/LyceumRegistry
(hebi) pkg> instantiate
(hebi) pkg> resolve
```

# Testing

`testsysid` function currently compares two models after setting one to have
 the wrong damping values for two joints.

 It correctly recovers the correct values.

```julia
julia> include("src/runhebi.jl")
julia> h = HebiPickup()
julia> h2 = HebiPickup()
julia> a = getsincontrols(h, 2000)
julia> result = testsysid(h, h2, a)
julia> result.minimizer
i2-element Array{Float64,1}:
 0.5001817850595129
 0.49973369119618827
```


The `gradsysid` function is an example of LBFGS optimization with a finite differenced
graddient function that uses multi-threading. It may be currently buggy.
