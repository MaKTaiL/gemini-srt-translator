from setuptools import setup, find_packages

setup(
    name="gemini-srt-translator",
    version="1.5.1",
    packages=find_packages(),
    install_requires=[
        "google-generativeai==0.8.4",
        "srt==3.5.3",
        "json-repair==0.39.1"
    ],
    author="Matheus Castro",
    description="A tool to translate subtitles using Google Generative AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maktail/gemini-srt-translator",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires='>=3.9',
)