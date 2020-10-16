## agnTk
**agntk** stands for **A**ctive **G**alactic **N**ucleus (AGN) **T**ime-Series Analysis Tool**k**it. It is a toolkit for conducting AGN time-series/variability analysis, mainly utilizing the continuous-time auto-regressive moving average model (CARMA)

<span style='color:red'>__Note:__</span> The notebooks associated with the code development have been moved to the `dev` branch. 

### Installation
pip distribution will be available soon. At this moment, you need to clone this repo and perform install with:
```
pip install -e .
```

### Development
`poetry` is used to solve dependencies and to build/publish this package. Below shows how setup the environment for development (assuming you already have `poetry` installed on your machine). 

1. Clone this repository, and enter the repository folder.
2. Create a python virtual environment and activate it. 
    ```
    python -m venv env
    source env/bin/activate
    ```
3. Install dependencies and **agnTk** in editable mode.
   ```
   poetry install
   ```

Now you should be ready to start adding new features. Be sure to checkout the normal practice regarding how to use `poetry` on its website. When you are ready to push, also make sure the poetry.lock file is checked-in if any dependency has changed. 