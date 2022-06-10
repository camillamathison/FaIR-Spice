from setuptools import setup

setup(
    name='fair-spice',
    version='0.0.1',
    packages=['fair_spice'],
    package_dir={"": "src"},
    install_requires=[
        'fair',
        'netcdf4',
        'importlib-metadata; python_version == "3.8"',
    ],
)