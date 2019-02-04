from setuptools import setup, find_packages

import os
import re

def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


def find_version(file):
    content = read_file(file)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content,
                              re.M)
    if version_match:
        return version_match.group(1)

requirements = resolve_requirements(os.path.join(os.path.dirname(__file__),
                                                 'requirements.txt'))

readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))
license = read_file(os.path.join(os.path.dirname(__file__), "LICENSE"))
shapenet_version = find_version(os.path.join(os.path.dirname(__file__), "shapenet",
                                           "__init__.py"))

setup(
    name='shapenet',
    version=shapenet_version,
    packages=find_packages(),
    url='https://github.com/justussschock/shapenet',
    author='Justus Schock',
    author_email='justus.schock@rwth-aachen.de',
    description='',
    test_suite="pytest",
    long_description=readme,
    license=license,
    install_requires=requirements,
    tests_require=["pytest-cov"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ]
)
