# coding: utf-8

"""
Jiaozifs Dataloader
"""

from setuptools import setup, find_packages

NAME = "jz-dataloader"
VERSION = "0.0.1"
# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools


setup(
    name=NAME,
    version=VERSION,
    description="jiaozifs dataloader for pytorch",
    author_email="",
    url="",
    keywords=["jiaozifs", "pytorch", "dataloader"],
    packages=find_packages(),
    include_package_data=True,
)
