"""Setup configuration for Dean package."""

from setuptools import setup, find_packages

setup(
    name="dean",
    version="1.0.0",
    description="Enterprise Meta Ads Automation System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Dean Team",
    python_requires=">=3.9",
    packages=find_packages(where=".", include=["src*"]),
    package_dir={"": "."},
    install_requires=[
        "facebook-business==19.0.1",
        "python-dotenv==1.0.1",
        "PyYAML==6.0.2",
        "requests==2.32.3",
        "pytz>=2020.1,<2024",
        "pandas==2.2.2",
        "openpyxl==3.1.5",
        "SQLAlchemy==2.0.35",
        "prometheus-client==0.20.0",
        "jsonschema==4.23.0",
        "supabase>=2.5.0",
        "schedule==1.2.0",
        "numpy>=1.24.0",
        "openai>=1.0.0",
        "sentence-transformers>=2.2.0",
        "redis>=5.0.0",
        "psutil>=5.9.0",
        "Pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "dean=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

