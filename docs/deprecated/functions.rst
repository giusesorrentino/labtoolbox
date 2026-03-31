=================
Removed Functions
=================

Legacy Statistical Functions
----------------------------

samples()
~~~~~~~~~

.. versionremoved:: 3.1.0
    The ``samples()`` function has been removed. Use ``scipy.stats`` distributions instead.

    **Migration:**

    .. code-block:: python

        # Old (removed)
        from labtoolbox.stats import samples
        data = samples(1000, 'normal', mu=0, sigma=1)

        # New (recommended)
        from scipy import stats
        data = stats.norm.rvs(loc=0, scale=1, size=1000)

noise()
~~~~~~~

.. versionremoved:: 3.1.0
    The ``noise()`` function has been removed. Use ``numpy.random`` instead.

    **Migration:**

    .. code-block:: python

        # New (recommended)
        import numpy as np
        noise = np.random.normal(0, 1, size=1000)

analyze_residuals()
~~~~~~~~~~~~~~~~~~~

.. versionremoved:: 3.1.0
    The ``analyze_residuals()`` function has been removed. Use ``stats.residuals()`` instead.

remove_outliers()
~~~~~~~~~~~~~~~~~

.. versionremoved:: 3.1.0
    The ``remove_outliers()`` function has been removed from ``labtoolbox.stats``.

Legacy Fitting Functions
------------------------

bootstrap_fit()
~~~~~~~~~~~~~~~

.. versionchanged:: 3.1.0
    Deprecated in favor of :func:`scipy.stats.bootstrap`.

    **Migration:**

    .. code-block:: python

        # Old (deprecated)
        from labtoolbox.fit import bootstrap_fit
        result = bootstrap_fit(...)

        # New (recommended)
        from scipy import stats
        result = stats.bootstrap(...)

Removed Signal Functions
------------------------

dfs()
~~~~~

.. versionremoved:: 3.1.0
    The ``dfs()`` function has been removed from ``labtoolbox.signals``.

harmonic()
~~~~~~~~~~

.. versionremoved:: 3.1.0
    The ``harmonic()`` function has been removed from ``labtoolbox.signals``.

decompose()
~~~~~~~~~~~

.. versionremoved:: 3.1.0
    The ``decompose()`` function has been removed from ``labtoolbox.signals``.
