.. :changelog:
0.4.4 (2025-06-24)
+++++++++++++++++++++
- Drop support for Python 3.10
- Drop support for Numpy < 2.0

0.4.4 (2025-06-24)
+++++++++++++++++++++
- Drop support for Python 3.8
- Add MJD support to `gpSimByTime`
- **Bug fixes:** #78, #79, #88
poetry run nox
0.4.3 (2023-12-15)
+++++++++++++++++++++
- Drop support for Python 3.7
- Bump `numba` requirement to `>=0.57.0`.
- **New Features:** Added seed options to `gpSimRand`, `gpSimFull`, and `addNoise`
- **Bug fixes:** #74, #75

0.4.1 (2023-06-12)
+++++++++++++++++++++
- Update reference to numpy bool/complex (#71)
- **Bug fixes:** #50, #54, #59

0.4.0 (2021-07-19)
+++++++++++++++++++++
- Fitting functions (i.e., `drw_fit`) are now fully modular (allow user provided optimization function, prior function and etc.)
- A new `addNoise` function to simulated random noise given measurement errors.
- **Bug fixes:** #44
- **API changes:** `n_iter` -> `n_opt` in fitting functions.

0.3.0 (2021-01-07)
+++++++++++++++++++++

- update parameter initialization in fit functions; removed `de` option #26, #27
- add few utils functions #30, #25
- add mcmc module #29
- ts simulation now support linear error
- added online documentation

0.2.3 (2020-12-08)
++++++++++++++++++

- add methods to CARMA_term conversion between CARMA and poly space
- fixed bugs and add tests for model 2nd order stat functions
- close #2, close #10

0.2.1 (2020-12-05)
++++++++++++++++++

- A bunch bug fixes in the ts.carma module
- Improved _min_opt optimizer, now added to all fitting functions
- Now using minimizer instead of differential evolution may result in more robust parameter estimates.

0.2.0 (2020-12-03)
++++++++++++++++++
Fixed some bugs and added new features.

- Fixed the instability issue when fitting time series to models higher than DHO/CARMA(2,1)
- Cleaned up the plotting module
- Added PSD, ACVF, and SF functions

0.1.0 (2020-11-09)
++++++++++++++++++
First release!

- Fully working CARMA kernels: `DRW_term`, `DHO_term` and `CARMA_term`
- Functions to simulate CARMA time series given a kernel
- Functions to fit arbitrary time series to CARMA models (still having instability issues with CARMA models higher than DHO/CARMA(2,1))
