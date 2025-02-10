from setuptools import setup, find_packages

setup(
    name="hep_rewgt_tk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "xgboost",
        "torch",
        "scikit-learn",
        "matplotlib"
    ],
    author="Yuntong Zhou",
    author_email="yuntongzhou@outlook.com",
    description="A package for reweighting in High Energy Physics using ML techniques",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JoyYTZhou/HEPReWgtTK",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)