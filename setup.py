from setuptools import find_packages, setup

curr_version = '0.1.0'

setup(
    name='shapenet',
    version=curr_version,
    packages=find_packages(),
    url='https://github.com/justusschock/shapenet',
    license='MIT',
    author='Justus Schock',
    author_email='justus.schock@rwth-aachen.de',
    install_requires=["pyyaml",
                      "torch>=1.0",
                      "torchvision",
                      "visdom",
                      "kaggle",
                      "shapedata@git+https://github.com/justusschock/shapedata.git#egg=shapedata-0.1",
                      "delira@git+https://github.com/justusschock/delira.git#egg=delira-0."],
    python_requires=">3.5",
    description='',
    entry_points={
        'console_scripts': [
            'export_to_jit=shapenet.scripts.export_to_jit:main',
            'predict_from_net=shapenet.scripts.predict_from_net:predict',
            "prepare_all_data=shapenet.scripts.prepare_datasets:prepare_all_data",
            "prepare_cat_dset=shapenet.scripts.prepare_datasets:prepare_cat_dset",
            "prepare_helen_dset=shapenet.scripts.prepare_datasets:prepare_helen_dset",
            "prepare_lfpw_dset=shapenet.scripts.prepare_datasets:prepare_lfpw_dset",
            "train_shapenet=shapenet.scripts.train_single_shapenet:train_shapenet",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ]
)
