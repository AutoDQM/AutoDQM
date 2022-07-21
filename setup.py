#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
from setuptools import setup, find_packages


with open('README.rst') as readme_file:
    readme = readme_file.read()

def get_version():
    _globals = {}
    with open(os.path.join("autodqm", "version.py")) as version_file:
        exec(version_file.read(), _globals)
    return _globals["__version__"]

requirements = ['lxml==4.6.5','requests-futures==1.0.0']

setup(
    author="Rob White",
    author_email='robert.stephen.white@cern.ch',
    classifiers=[
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=requirements,
    description="Statistical analysis toolkit for DQM of histograms",
    long_description=readme,  # + '\n\n' + history,
    include_package_data=True,
    keywords=['ROOT', 'DQM', 'analysis', 'particle physics', 'HEP'],
    name='AutoDQM',
    url='https://github.com/GluonicPenguin/AutoDQM',
    zip_safe=True,
)
