from setuptools import setup, find_packages

setup(
    name="gemini-srt-translator",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "google-generativeai==0.8.3",
        "srt==3.5.3",
    ],
    author="Matheus Castro",
    description="A tool to translate subtitles using Google Generative AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maktail/gemini-srt-translator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)