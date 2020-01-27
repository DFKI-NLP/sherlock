from io import open
from setuptools import find_packages, setup


extras = {}

extras["torch"] = ["torch"]

extras["serving"] = ["pydantic", "uvicorn", "fastapi"]
extras["all"] = extras["serving"] + ["torch"]

extras["testing"] = ["pytest", "pytest-xdist"]
extras["quality"] = ["black", "isort", "flake8", "mypy"]
extras["docs"] = ["recommonmark", "sphinx", "sphinx-markdown-tables", "sphinx-rtd-theme"]
extras["dev"] = extras["testing"] + extras["quality"] + ["torch"]

setup(
    name="sherlock",
    version="0.0.1a",
    author="Christoph Alt",
    author_email="christoph.alt@posteo.de",
    description="Sherlock - A state-of-the-art information extraction framework",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="neural information extraction named entity recognition relation extraction entity linking event extraction",
    license="Apache",
    url="https://github.com/ChristophAlt/sherlock",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=["transformers", "spacy", "seqeval", "registrable"],
    extras_require=extras,
    entry_points={
        "console_scripts": []
    },
    python_requires='>=3.7.0',
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
