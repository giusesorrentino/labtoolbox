===============
Removed Modules
===============

Legacy Fitting Module
---------------------

fit
~~~

.. versionremoved:: 3.1.0
    The ``labtoolbox.fit`` module has been removed from the public API.

    **Migration:**

    - Use :mod:`labtoolbox.stats` for ``lin_fit`` and ``model_fit``.
    - Use :func:`scipy.stats.bootstrap` instead of ``bootstrap_fit``.

Legacy Uncertainty Module
-------------------------

uncertainty
~~~~~~~~~~~

.. versionremoved:: 3.1.0
    The ``labtoolbox.uncertainty`` module has been removed from the public API.

    **Migration:**

    - Use :func:`labtoolbox.stats.propagate` for uncertainty propagation.
