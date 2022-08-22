# numerical_gill

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> numerical_gill

It is authored by Paul Keil.

To (locally) reproduce this project, do the following:

0. Download this code base.

1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the showcase notebook.
You can also feed the model with real data to use as forcing.
