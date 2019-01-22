****************
ForeGroundBuster
****************
Parametric component separation for Cosmic Microwave Background observations
############################################################################

This library provides easy-to-use functions to perform

* separation of frequency maps into component maps 
  `(Stompor et al. 2009) <https://academic.oup.com/mnras/article/392/1/216/1071929>`_
* forecasts of component separation
  
  * when the model is correct
    `(Errard et al. 2011) <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.84.069907>`_
  * when the model is incorrect
    `(Stompor et al. 2016) <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.94.083526>`_

Install
#######

.. code-block:: bash

    git clone git@github.com:fgbuster/fgbuster.git
    cd fgbuster
    pip install -e .

In the last line you might need ``--user``.  Using ``-e`` is not necessary, but
it makes latest changes available to you every time you ``git pull``.

Usage
#####
The most common case of use are covered in the :ref:`examples`.

Support
#######

If you encounter any difficulty in installing and using the code or you think
you found a bug, please `open an issue
<https://github.com/fgbuster/fgbuster/issues/new>`_.

Contributing
############

Development takes place on `github
<https://github.com/fgbuster/fgbuster>`_.
See  :ref:`contributing`.

----

Acknowledgements
----------------

Davide Poletti acknowledges the support from the
`COSMOS network <http://www.cosmosnet.it>`_ of the Italian Space Agency, the
`RADIOFOREGROUNDS <http://www.radioforegrounds.eu/>`_ project funded by the
European Commissions H2020 Research Infrastructures under the Grant Agreement
687312, and the
`International School for Advanced Studies <http://www.sissa.it>`_ (SISSA)

Josquin Errard acknowledges support of the French National Research Agency
(Agence National de Recherche) grant, ANR BxB.

|sissa| |radioforegrounds| |cosmos|

.. |sissa| image:: /_static/logo_sissa.png
    :alt: SISSA
    :height: 100px
    :target: http://www.sissa.it

.. |radioforegrounds| image:: /_static/logo_radioforegrounds_v.png
    :alt: RADIOFOREGROUNDS
    :height: 100px
    :target: http://www.radioforegrounds.eu

.. |cosmos| image:: /_static/logo_cosmos.png
    :alt: COSMOS
    :height: 100px
    :target: http://www.cosmosnet.it
