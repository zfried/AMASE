from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("amase/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="amase",
    version="0.1.0",
    author="Zachary Fried",
    author_email="zfried@mit.edu",
    description="AMASE Mixture Analysis Algorithm for rotational spectroscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zfried/AMASE",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
)
