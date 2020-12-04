## v0.2 (2020-12-03)
---
Fixed some bugs and added new features.

- Fixed the instability issue when fitting time series to models higher than DHO/CARMA(2,1)
- Cleaned up the plotting module
- Added PSD, ACVF, and SF functions


## v0.1 (2020-11-09)
---
First release!

### Features:
- Fully working CARMA kernels: `DRW_term`, `DHO_term` and `CARMA_term`
- Functions to simulate CARMA time series given a kernel
- Functions to fit arbitrary time series to CARMA models (still having instability issues with CARMA models higher than DHO/CARMA(2,1))
  