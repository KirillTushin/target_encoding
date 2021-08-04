"""Setup for target encoding module"""

from os.path import join, dirname

from setuptools import setup, find_packages

from target_encoding import __version__

with open(join(dirname(__file__), 'README.md')) as readme_file:
    LONG_DESCRIPTION = readme_file.read()

with open('requirements.txt') as requirements_file:
    INSTALL_REQUIRES = '\n'.join(requirements_file.readlines())

NAME = 'target_encoding'
VERSION = __version__
PACKAGES = find_packages()
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
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    install_requires=INSTALL_REQUIRES,
    test_suite=TEST_SUITE,
    classifiers=CLASSIFIERS,
    include_package_data=True,
)
