# coding: utf-8

"""
Jiaozifs Dataloader
"""

from setuptools import setup, find_packages  # noqa: H301

NAME = "jiaozifs-dataloader"
VERSION = "0.0.1"
# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

REQUIRES = ["urllib3 >= 1.15", "six >= 1.10", "certifi", "python-dateutil"]

setup(
    name=NAME,
    version=VERSION,
    description="jiaozifs dataloader for pytorch",
    author_email="",
    url="",
    keywords=["jiaozifs", "pytorch", "dataloader"],
    install_requires=REQUIRES,
    packages=find_packages(),
    include_package_data=True,
)
