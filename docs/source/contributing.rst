.. _contributing:

************
Contributing
************

Follow the `Google style guide
<https://github.com/google/styleguide/blob/gh-pages/pyguide.md>`_
and...write docstrings! They auto-generate the documentation. Use the 
`numpy sphinx syntax <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.

Generating the documentation
----------------------------
Call ``ROOT_DIR`` the folder where you cloned the FGBuster repository.
Clone the documentation branch

.. code-block:: bash

   cd ROOT_DIR
   mkdir fgbuster-docs
   cd fgbuster-docs
   git clone git@github.com:fgbuster/fgbuster.git html
   cd html
   git checkout gh-pages

You have now two clones of the FGBuster repository. It is important that they
have the right relative location.

To check the documentation locally, open ``ROOT_DIR/fgbuster-docs/html/index.html``
with your browser.

Whenever you make changes to FGBuster, build the documentation with

.. code-block:: bash

   cd ROOT_DIR
   cd fgbuster/docs/
   make html

and refresh your browser. The documentation is up-to-date.

To update the online documentation,

.. code-block:: bash

   cd ROOT_DIR
   cd fgbuster-docs/html
   git add --all
   git commit
   git push
