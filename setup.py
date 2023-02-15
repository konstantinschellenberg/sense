# -*- coding: UTF-8 -*-

"""
This file is part of SenSE.
(c) 2023- Thomas Weiß, Alexander Löw
For COPYING and LICENSE details, please refer to the LICENSE file
"""

from __future__ import absolute_import
from __future__ import print_function

from setuptools import setup
from setuptools import find_packages

import io

from os.path import dirname
from os.path import join


__version__ = None
with open('sense/version.py') as f:
    exec(f.read())

def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()

with open('docs/requirements.txt') as ff:
    required = ff.read().splitlines()

setup(name='sense',
      version=__version__,
      description='SenSE - Community SAR ScattEring model ',
      long_description=read('README.md'),
      license='GNU license',
      author='Thomas Weiß, Alexander Löw',
      author_email='weiss.thomas@lmu.de',
      url='https://github.com/McWhity/sense',
      packages=['sense','sense.surface','sense.dielectric'],
      install_requires=required,
      package_data={},
      include_package_data=True,
      zip_safe=False,
)
