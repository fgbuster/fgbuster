# DustBuster
Parametric component separation for Cosmic Microwave Background observations

## Install
The code is quickly evolving. Install it with

```bash
git clone git@github.com:dpole/xForecast.git
cd xForecast
[sudo] python setup.py develop [--user]
```

so that you can keep up with the latest changes simply with `git pull`.

You can now access the routines from anywhere with

```python
import fgbuster
```

## Usage
The library provides utilities for several case of use.
The `example` folder will contain notebooks that illustrate them.
Currently, it contains an example of map-based component separation.

## Note
If you have access to this repo it means that you are either a developer
or a test user. In the latter case, please do not hesitate to
 - make suggestion
 - report bugs
 - propose new features

and keep in mind that
 - we will soon move from the old `xForecast` repo to final repo and organization (you will be notified)
 - API may change
 - what routines do may change

We do our best to keep docstrings up to date
