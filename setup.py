import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jaxdsp",
    version="0.3.1",
    author="Karl Hiner",
    author_email="karl.hiner@gmail.com",
    description="Fast, differentiable audio processors on the CPU or GPU, with a browser client for real-time control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khiner/jaxdsp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    # install_requires=['jaxlib', 'jax[cpu]'],
    install_requires=['jax-spectral @ git+https://github.com/cifkao/jax-spectral.git@43085eb9b094145cd15b559c6c882fe162d75a01'],
)
