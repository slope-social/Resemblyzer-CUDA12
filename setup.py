from setuptools import setup, find_packages

setup(
    name="resemblyzer",
    version="0.2.5",  # Incrementing version for CUDA 12.1 nightly support
    description="Real Time Voice Cloning with CUDA 12 Support",
    url="https://github.com/slope-social/Resemblyzer-CUDA12",
    author="Slope Social",
    author_email="hey@slope.social",
    packages=find_packages(),
    package_data={
        "resemblyzer": ["pretrained.pt"],
    },
    python_requires=">=3.10",
    install_requires=[
        "torch==2.6.0.dev20241026+cu121",
        "torchaudio==2.5.0.dev20241026+cu121",
        "librosa>=0.10.1",
        "numpy>=1.23.5",
        "webrtcvad>=2.0.10",
        "scipy>=1.14.1",
        "typing_extensions>=4.12.0"
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/nightly/cu121"
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech"
    ]
)
