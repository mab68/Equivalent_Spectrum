# **Equivalent Spectrum**

Structure function calculation is availble via `mpi_sf.py` with naively implemented multi-processing.

Equivalent spectrum calculation (both "uncorrected" and "debiased") and additional required/recommended functions are supplied in `equiv_spectrum.py`.

See usage examples in Jupyter notebooks: `example_1d.ipynb` ($D=1$) and `example_2d.ipynb` ($D=2$). The required data (fractional Brownian motion fields) are provided in the `example_data/` folder.

## Requirements
```
numpy
scipy
sympy
```
For the multi-processing structure function calculation
```
mpi4py
```
