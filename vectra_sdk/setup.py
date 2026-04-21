from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vectra-sdk",
    version="0.1.0",
    author="Rohit Gomes",
    author_email="gomesrohit92@gmail.com",
    description="A high-level inference SDK for Vectra Engine few-shot models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["vectra", "vectra.*"]),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "Pillow",
        "numpy",
        "opencv-python",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
