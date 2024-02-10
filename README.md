# Publicly available scripts associated with Leung et al. 2023
Link to paper: https://doi.org/10.1093/mnras/stae225

This repository contains 4 scripts in the folder `scripts`. Three of which are used to reproduce figures 7, 8 and 9 of the paper, while the last script is a demonstration of how to use `Bagpipes` to perform one of the fits following the method detailed in the paper.

Some of the data required for these scripts are not stored here. They can be found at https://doi.org/10.17630/ac0b406c-1c59-41e6-8b73-026790a0c1ca. After downloading files from there, the file struction should look like the following:
```
.
├── data
│   ├── peng2015                        # data from Peng et al. 2015 MZ relation
│   ├── stacked_spectrum                # directory where all the stacked spectra and additional masking files live
│   ├── posterior_percentiles.csv       # table of the 16th, 50th and 84th percentiles of a list of fitted properties of the PSB sample
│   ├── posterior_samples.csv           # table of the individual posterior draw values of a list of fitted properties of the PSB sample
│   └── skylines.txt                    # table of skyline wavelength, width and flux, adopted from Hanuschik 2003 https://ui.adsabs.harvard.edu/abs/2003A%26A...407.1157H/abstract
├── pipes                               # Output files from bagpipes
├── plots                               # plots saved by scripts in this repository
└── scripts                             # The four scripts
    ├── code_bits                       # a collection of .py files that adds additional functionalities to the base bagpipes build developed for this study
    ├── fit_functions_demo.ipynb
    ├── fig7_example_fitting_plot.ipynb
    ├── fig8_metallicity_burst_vs_old.ipynb
    └── fig9_MZ_relation.ipynb
```
