import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'fgbuster',
    version = '1.1.0',
    author = 'Davide Poletti, Josquin Errard and the FGBuster developers',
    author_email = 'davide.pole@gmail.com, josquin@apc.in2p3.fr',
    description = ('Handy parametric component separation tools'),
    license = 'GPLv3',
    keywords = 'statistics cosmology cmb foregrounds',
    url = 'https://github.com/fgbuster/fgbuster',
    packages = ['fgbuster'],
    package_data={'': ['fgbuster/templates/*.fits']},
    include_package_data=True,
    long_description = read('README.rst'),
    install_requires = [
        'parameterized',
        'corner',
        'numdifftools',
        'sympy',
        'healpy',
    ],
    test_suite = 'fgbuster.test'
)
