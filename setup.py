import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jaxdsp",
    version="0.2.0",
    author="Karl Hiner",
    author_email="karl.hiner@gmail.com",
    description="Fast, differentiable audio processors on the CPU or GPU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khiner/jaxdsp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
