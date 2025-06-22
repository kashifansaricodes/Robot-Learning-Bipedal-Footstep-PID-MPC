from setuptools import setup, find_packages

setup(

    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorflow==1.15",
        "kinpy",
        "protobuf==3.20.*",
        "mpi4py"
    ],
    python_requires="<3.8",
)
