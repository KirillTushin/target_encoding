"""Setup for target encoding module"""

from os.path import join, dirname

from setuptools import setup, find_packages

from target_encoding import __version__

with open(join(dirname(__file__), 'README.md')) as readme_file:
    LONG_DESCRIPTION = readme_file.read()

setup(
    name='target_encoding',
    version=__version__,
    packages=find_packages(),
    author='Tushin Kirill',
    author_email='kirya.tushin1@yandex.ru',
    include_package_data=True,
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    install_requires=[
        'numpy==1.16.2',
        'scikit-learn==0.20.3',
    ],
    url="https://github.com/KirillTushin/target_encoding",
    test_suite='tests',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

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
