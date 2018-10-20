from setuptools import setup, find_packages

setup(
    name="simulation_based_calibration",
    version="0.0.1",
    description='PyMC3 implementation of "Simulation Based Calibration"',
    author="Colin Carroll",
    url="http://github.com/colcarroll/simulation_based_calibration",
    packages=find_packages(),
    install_requires=["pymc3", "tqdm", "matplotlib", "numpy"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
)
