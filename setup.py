import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = []
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="ctrl4ai",
    version="1.0.5",
    author="Shaji James",
    author_email="shajijames7@gmail.com",
    description="A helper package for Machine Learning and Deep Learning Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vkreat-tech/ctrl4ai",
    include_package_data=True,
    packages=setuptools.find_packages(),
    #package_data={"": ["*.txt"]},
    #exclude_package_data={"": ["*.csv.gz"]},
    data_files=[('dictionary', ['ctrl4ai\dictionary\likert_scales.txt'])],
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
)