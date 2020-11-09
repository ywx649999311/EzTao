![tests](https://github.com/ywx649999311/EzTao/workflows/tests/badge.svg)
# EzTao (易道)
**EzTao** is a toolkit for conducting AGN time-series/variability analysis, mainly utilizing the continuous-time auto-regressive moving average model (CARMA)

## Installation
```
pip install eztao
```

#### Dependencies
>```
>python = "^3.7"
>numpy = "^1.19.1"
>celerite = "^0.4.0"
>matplotlib = "^3.3.1"
>scipy = "^1.5.2"
>numba = "^0.51.2"
>```

## Development
`poetry` is used to solve dependencies and to build/publish this package. Below shows how setup the environment for development (assuming you already have `poetry` installed on your machine). 

1. Clone this repository, and enter the repository folder.
2. Create a python virtual environment and activate it. 
    ```
    python -m venv env
    source env/bin/activate
    ```
3. Install dependencies and **EzTao** in editable mode.
   ```
   poetry install
   ```

Now you should be ready to start adding new features. Be sure to checkout the normal practice regarding how to use `poetry` on its website. When you are ready to push, also make sure the poetry.lock file is checked-in if any dependency has changed. 
