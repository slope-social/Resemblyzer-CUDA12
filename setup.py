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
        "librosa>=0.9.1",
        "numpy>=1.20.0",
        "webrtcvad>=2.0.10",
        "torch>=2.0.0",
        "scipy>=1.2.1",
        "typing",
    ],
    include_package_data=True
)
