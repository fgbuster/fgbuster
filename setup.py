import os
from setuptools import setup, find_packages

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'fgbuster',
    version = '1.0.0',
    author = 'Davide Poletti, Josquin Errard and the FGBuster developers',
    author_email = 'davide.pole@gmail.com, josquin@apc.in2p3.fr',
    description = ('Handy parametric component separation tools'),
    license = 'GPLv3',
    keywords = 'statistics cosmology cmb foregrounds',
    url = 'https://github.com/fgbuster/fgbuster',
    packages = find_packages(),
    include_package_data=True,
    long_description = read('README.rst'),
    install_requires = [
        'parameterized',
        'corner',
        'numdifftools',
        'sympy',
        'healpy',
        'pysm @ git+https://github.com/bthorne93/PySM_public.git@master#egg=pysm',
        'setuptools_git'
    ],
    test_suite = 'fgbuster.test'
)
