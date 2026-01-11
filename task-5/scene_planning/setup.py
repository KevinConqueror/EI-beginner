from setuptools import setup, find_packages

setup(
    name="embodiment",
    version="1.2.6",
    packages=find_packages(where="python_package"),
    package_dir={"": "python_package"},
    install_requires=[],
    description="Embodiment",
)