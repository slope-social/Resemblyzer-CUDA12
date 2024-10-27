from setuptools import setup, find_packages

setup(
    name="resemblyzer",
    version="0.2.3",
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
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "librosa>=0.10.1",
        "numpy>=1.23.5",
        "webrtcvad>=2.0.10",
        "scipy>=1.14.1",
        "typing_extensions>=4.12.0"
    ],
    include_package_data=True
)
