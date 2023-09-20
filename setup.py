# -*- coding: utf-8 -*-
import os
import re

from setuptools import find_packages, setup

path = os.path.abspath(os.path.dirname(__file__))
readme = open(os.path.join(path, "README.md")).read()

try:
    with open(os.path.join(path, "nfsdata", "__init__.py"), "r") as f:
        contents = f.read()
    version = re.search('__version__ = "(.*)"', contents).groups()[0]
except Exception:
    version = ""

setup(
    name="nfsdata",
    version=version,
    description="utilities for working with nuclear forensics datasets",
    long_description="\n\n".join(readme),
    author="Jakob Johnson",
    author_email="",
    url="https://github.com/jakobottar/nfs-dataset",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["ml-pyxis", "pandas", "torch", "Pillow", "numpy", "tifffile", "PyYAML"],
)
