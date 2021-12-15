from io import open
from setuptools import find_packages, setup


extras = {}

extras["torch"] = ["torch"]

extras["serving"] = ["pydantic", "uvicorn", "fastapi"]
extras["all"] = extras["serving"] + ["torch"]

extras["testing"] = ["pytest", "pytest-xdist"]
extras["quality"] = ["black==19.10b0", "isort>=4.3.21,<5.0.0", "flake8~=3.8", "mypy"]
extras["docs"] = ["recommonmark", "sphinx", "sphinx-markdown-tables", "sphinx-rtd-theme"]
extras["dev"] = extras["testing"] + extras["quality"] + ["torch"]

setup(
    name="sherlock",
    version="0.0.2",
    author="Christoph Alt,Leonhard Hennig,Marc Hübner,Gabriel Kressin",
    author_email="christoph.alt@posteo.de,marc.huebner@dfki.de,leonhard.hennig@dfki.de,gabriel.kressin@dfki.de",
    description="Sherlock - A state-of-the-art information extraction framework",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="neural information extraction named entity recognition relation extraction entity linking event extraction",
    license="Apache",
    url="https://github.com/DFKI-NLP/sherlock",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=[
        # core
        "spacy>=2.1.0,<3.3",
        "transformers>=4.1,<4.13",
        "seqeval",
        "torch>=1.6.0,<1.11.0",
        "registrable",
        "tensorboardX>=1.2",
        "allennlp==2.8.0",

        # Microscope
        "jsonnet>=0.10.0 ; sys.platform != 'win32'",
        "flask",
        "flask_cors",
        "gevent",
    ],
    extras_require=extras,
    include_package_data=True,  # include all files specified in the MANIFEST.in file
    entry_points={
        "console_scripts": []
    },
    python_requires='>=3.7.1',
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
