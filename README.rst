===============================
Lichens for Analyzing XENON1T
===============================

.. image:: https://api.codacy.com/project/badge/Grade/724ba633bd6b4079b977e0aa623b327d
     :target: https://www.codacy.com/app/tunnell/lax?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=XENON1T/lax&amp;utm_campaign=Badge_Grade
     :alt: Style


This package contains the stanardized event selections for XENON1T analysis.


Documentation
--------------

Code organization
==================

Lax stores event selections in classes called "Lichens". These are organized in three different files:

* `sciencerun0`: selections for XENON1T's first (short) science run (SR0);
* `sciencerun1`: selections for XENON1T's main science run (SR1);
* `postsr1`: improved versions of cuts applicable to the main science run (SR1) and/or later datasets.

The selections in the `sciencerunX` files are exactly as they were used in analyses supporting XENON1T's main science paper (`Phys. Rev. Lett. 121, 111302 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.111302>`_ / `arXiv 1805.12562  <https://arxiv.org/abs/1805.12562>`_)

Some lichens in these files are 'summary lichens', containing


Applying selections
======================
The easiest way to apply selections is to use `hax.cuts.apply_lichen`, see the `second hax tutorial <https://github.com/XENON1T/hax/blob/master/examples/02_getting_serious.ipynb>`_ for more information. Alternatively, you can annotate the events (see below), then cut on the newly created columns.

Some lichens rely on the presence of non-standard variables in your event dataframe. This is usually indicated in the lichen docstrings. The default cut sets should work from the default set of minitrees


Annotating events
=======================
For some studies, you may wish to know which events would pass which selections. For this, use the `process` method of the individual lichen classes. For an example, see `here <https://github.com/XENON1T/lax/blob/master/examples/test_postsr1.ipynb>`_.



Credits
---------

Many XENON analysts contributed one or more event selections to lax: please see the lichen source files for details.

Development lead: Christopher Tunnell

Maintainers:

* Christopher Tunnell (SR0)
* Shayne Reichard (SR1, post-SR1)
* Patrick Deperio (SR1)
* Jelle Aalbers (post-SR1)
