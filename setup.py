from setuptools import find_packages, setup


with open("README.md") as f:
    long_description = f.read()

install_requires = [
    "wandb",
    "pandas",
    "opencv-python",
    "tqdm",
    "numpy",
    "torch",
    "torchvision",
]

if __name__ == "__main__":
    setup(
        name="sr_mobile_pytorch",
        description="An unofficial PyTorch port of NJU-Jet/SR_Mobile_Quantization.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="w11wo",
        author_email="wilson@bookbotkids.com",
        url="https://github.com/w11wo/sr_mobile_pytorch",
        license="Apache License",
        packages=find_packages(),
        install_requires=install_requires,
        include_package_data=True,
        platforms=["linux", "unix", "windows"],
        python_requires=">=3.6",
    )
