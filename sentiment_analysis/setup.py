from setuptools import setup, find_packages

setup(
    name="sentiment-analysis",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "fastapi",
        "uvicorn",
    ],
    extras_require={
        "test": [
            "pytest",
            "httpx2",
            "httpx"
        ],
    },
    scripts={
        "sentiment-analysis": "src.sentiment.cli:main"
    },
)
