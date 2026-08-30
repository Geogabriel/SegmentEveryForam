import setuptools

long_description = """\
'segmenteveryforam' is a Python package for automated segmentation
and analysis of foraminifera in images.

SegmentEveryForam is adapted from the Segmenteverygrain package
developed by Zoltan Sylvester and extends the workflow toward
foraminiferal image segmentation and analysis.
"""

setuptools.setup(
    name="segmenteveryforam",
    version="0.1.0",
    author="Gabriel Ojo",
    description="A U-Net and SAM-based package for segmenting foraminifera in images",
    keywords="foraminifera, micropaleontology, image segmentation, morphometrics, segment anything model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Geogabriel/SegmentEveryForam",
    python_requires=">=3.10",
    packages=["segmenteveryforam"],
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pillow",
        "scikit-image",
        "tqdm",
        "opencv-python",
        "networkx",
        "rasterio",
        "shapely",
        "tensorflow",
        "sam2",
        "scikit-learn",
        "rtree",
        "pandas",
        "geopandas",
        "keras",
    ],
    license_files=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)