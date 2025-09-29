from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="omninumpy",
    version="0.1.0",
    author="OmniNumPy Contributors",
    author_email="omninumpy@example.com",
    description="A NumPy wrapper providing backward compatibility and backend switching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omninumpy/omninumpy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=2.0",
    ],
    extras_require={
        "torch": ["torch"],
        "cupy": ["cupy"],
        "jax": ["jax", "jaxlib"],
    },
    test_suite="omninumpy.tests",
)