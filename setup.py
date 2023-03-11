import os
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'varya',
    version = '0.0.2a1',
    author = 'Kritik Seth',
    author_email = 'sethkritik@gmail.com',
    description = 'Machine Learning Tools',
    py_modules = [''],
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/varya-ml/varya',
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