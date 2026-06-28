# **Equivalent Spectrum**

The code used in the paper: [Direct power spectral density estimation from structure functions without Fourier transforms](https://doi.org/10.1063/5.0310561)

**Please see my Python package [kea](https://github.com/mab68/kea) for an updated and detailed implementation.**

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

## Citation
If you make use of this code in a publication, please cite our accompanying paper:
> Mark A. Bishop, Sean Oughton, Tulasi N. Parashar, Yvette C. Perrott; Direct power spectral density estimation from structure functions without Fourier transforms. Physics of Fluids 1 February 2026; 38 (2): 025107. https://doi.org/10.1063/5.0310561
