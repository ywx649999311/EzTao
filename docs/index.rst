EzTao
=====
**EzTao** is a Python toolkit for conducting time-series analysis using continuous-time autoregressive moving average (CARMA) processes. It uses `celerite`_ (a fast gaussian processes regression library) to compute the likelihood of a set of proposed CARMA parameters given the input time series. Comparing to existing tools for performing CARMA analysis in Python which use *Kalman filter* to evaluate the likelihood function (e.g., `Kali`_), **EzTao** offers a more scalable solution (see the `celerite paper`_ for a comparison). 

**EzTao** consist of tools to both simulate CARMA processes and fit (maximum likelihood estimation or MLE) time series to CARMA models. The current version of **EzTao** is built on top of `celerite`_, future versions will take advantage of `celerite2`_ (still under active development) for a better integration with other probabilistic programing libraries such as `PyMC3`_. 

Installation
------------
**EzTao** can be installed with `pip`_ using::

   pip install eztao


.. raw:: html

    <br>

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   notebooks/00_About_CARMA
   notebooks/01_Quick_start

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   notebooks/02_Simulation
   notebooks/03_Fit
   notebooks/04_MCMC
   notebooks/05_OtherUtils

.. toctree::
   :maxdepth: 1
   :caption: Python API

   python/carma
   python/ts
   python/viz


Changelog
---------
.. include:: ../Changelog.rst

.. _celerite: https://celerite.readthedocs.io/en/stable/
.. _Kali: https://github.com/AstroVPK/kali
.. _celerite paper: https://iopscience.iop.org/article/10.3847/1538-3881/aa9332
.. _celerite2: https://github.com/exoplanet-dev/celerite2
.. _PyMC3: https://docs.pymc.io/
.. _pip: https://pip.pypa.io/en/stable/