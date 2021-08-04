"""Setup for target encoding module"""

import io
from os import path

from setuptools import setup, find_packages

from target_encoding import __version__


this_directory = path.abspath(path.dirname(__file__))
README_PATH = path.join(this_directory, 'README.md')
REQUIREMENTS_PATH = path.join(this_directory, 'requirements.txt')

with open(README_PATH, encoding='utf-8') as readme_file:
    LONG_DESCRIPTION = readme_file.read()

with io.open(REQUIREMENTS_PATH, 'r') as requirements_file:
    INSTALL_REQUIRES = requirements_file.read().splitlines()

NAME = 'target_encoding'
VERSION = __version__
PACKAGES = find_packages(exclude=("tests",))
URL = 'https://github.com/KirillTushin/target_encoding'
LICENSE = 'MIT'
AUTHOR = 'Tushin Kirill'
AUTHOR_EMAIL = 'tushin.ka@phystech.edu'
DESCRIPTION = 'Module for target encoding'
TEST_SUITE = 'tests'
CLASSIFIERS = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

setup(
    name=NAME,
    version=VERSION,
    packages=PACKAGES,
    url=URL,
    download_url=URL,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    install_requires=INSTALL_REQUIRES,
    test_suite=TEST_SUITE,
    classifiers=CLASSIFIERS,
    long_description_content_type='text/markdown',
    include_package_data=True,
)
