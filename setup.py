from setuptools import setup, find_packages

setup(
    name="cassie_rl_walking",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorflow==1.15",
        "kinpy",
        "protobuf==3.20.*",
        "mpi4py"
    ],
    author="Zhongyu Li",
    author_email="zhongyu_li@berkeley.edu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Creative Commons Attribution ShareAlike 4.0",
    ],
    python_requires="<3.8",
)
