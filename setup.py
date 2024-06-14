import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

deps = [
    "torch==1.13.0",
    "pandas==1.5.3",
    "numpy==1.24.1",
    "transformers==4.30.2",
    "sentencepiece==0.1.97",
    "evaluate==0.4.0",
    "datasets==2.14.6",
    "accelerate==0.20.1",
    "seqeval==1.2.2",
    "marisa-trie==1.0.0",
    "SPARQLWrapper==2.0.0",
    "invoke==1.7.0",
    "ipywidgets==8.0.4",
    "notebook==6.5.2",
    "black==23.3.0",
    "pylint==2.13.5",
    "python-dotenv==1.0.1",
]


setuptools.setup(
    name="sparkle",  # Replace with your own username
    version="0.0.1",
    description="End to End Natural language to SPARQL package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["sparkle"],
    install_requires=deps,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
)
