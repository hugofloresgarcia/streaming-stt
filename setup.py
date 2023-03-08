from setuptools import setup, find_packages


setup(
    name="streaming-stt",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "pyaudio",
        "google-cloud-speech",
        "google-cloud-storage",
        "tensorflow", 
        "tensorflow_hub",
        "numpy",
        "argbind"
    ]
)