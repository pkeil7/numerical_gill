# A numerical Gill Model to simulate large scale tropical circulation

This model is based on the works of Matsuno (1966) and Gill (1980). 3 prognostic variables `u`,`v` and `p` on a 2-dimensional spatial domain are integrated in time forced by a convective heating `Q`.

This code base is using the Julia Language (advisably at least 1.7.0) and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> numerical_gill


To (locally) reproduce this project, do the following:

0. Download this code base.

1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you.

Check out `notebooks/showcase.ipynb` for more information and a demonstration of how to reproduce the solution to idealised forcing from Gill 1980.


## References:

- Matsuno, Taroh. "Quasi-geostrophic motions in the equatorial area." Journal of the Meteorological Society of Japan. Ser. II 44.1 (1966): 25-43.
- Gill, Adrian E. "Some simple solutions for heat‐induced tropical circulation." Quarterly Journal of the Royal Meteorological Society 106.449 (1980): 447-462.
