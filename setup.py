from setuptools import setup, find_packages

setup(
    name="resemblyzer",
    version="0.2.1",
    description="Real Time Voice Cloning: Transfer learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis",
    url="https://github.com/slope-social/Resemblyzer-CUDA12",
    author="Slope Social",
    author_email="hey@slope.social",
    packages=find_packages(),
    package_data={
        "resemblyzer": ["pretrained.pt"],
    },
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.0.0+cu121",
        "torchaudio>=2.0.0+cu121",
        "numpy>=1.17.0",
        "scipy>=1.3.0",
        "scikit-learn>=0.22.0",
        "librosa>=0.8.0",
        "sounddevice>=0.4.0",
        "webrtcvad>=2.0.10",
    ],
    include_package_data=True
)
