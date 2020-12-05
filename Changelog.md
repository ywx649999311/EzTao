## v0.2.1 (2020-12-05)
---
- A bunch bug fixes in the ts.carma module
- Improved _min_opt optimizer, now added to all fitting functions
- Now using minimizer instead of differential evolution may result in more robust parameter estimates.

<br>

## v0.2.0 (2020-12-03)
---
Fixed some bugs and added new features.

- Fixed the instability issue when fitting time series to models higher than DHO/CARMA(2,1)
- Cleaned up the plotting module
- Added PSD, ACVF, and SF functions

<br>

## v0.1.0 (2020-11-09)
---
First release!

### Features:
- Fully working CARMA kernels: `DRW_term`, `DHO_term` and `CARMA_term`
- Functions to simulate CARMA time series given a kernel
- Functions to fit arbitrary time series to CARMA models (still having instability issues with CARMA models higher than DHO/CARMA(2,1))
  