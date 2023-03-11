import os
import setuptools
import varya

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'varya',
    version = varya.__version__,
    author = varya.__author__,
    author_email = 'sethkritik@gmail.com',
    description = 'Machine Learning Tools',
    py_modules = [''],
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = varya.__url__,
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Development Status :: 1 - Planning'
    ],
    packages = setuptools.find_namespace_packages(include=['varya', 'varya.*']),
    install_requires = [
    'pandas',
    'numpy',
    'torch'
      ],
    python_requires='>=3.6'
)