from setuptools import setup
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='nanodeep',
    version='0.0.3',
    packages=setuptools.find_packages(),
    url='https://github.com/shadow-lang',
    license='MIT License',
    author='Yusen Lin',
    author_email='1014903773@qq.com',
    description='A package used to achieve adaptive sample',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy >= 1.20.0',
        'torch >= 1.13.1',
        'torchmetrics == 0.9.0',
        'matplotlib >= 3.4.0',
        'ont-fast5-api >= 4.1.1',
        'scikit-learn >= 1.0',
        'tqdm >= 4.64.1',
        'PyYAML >= 6.0',
        'importlib-metadata >= 6.0.0',
        'importlib-resources >= 5.12.0',
        'grpcio >= 1.47.0',
        'protobuf >= 3.20.1',
        'pandas >= 1.1.0',
        'toml >= 0.10.2',
        'pyRFC3339 >= 1.1',
        'protobuf == 3.20.1',
        'ont-pyguppy-client-lib >= 6.4.2',
        'minknow-api >= 5.5.2'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',

    entry_points={
          'console_scripts': [
            'nanodeep_adaptivesample = nanodeep.nanodeep_adaptivesample:main',
            'nanodeep_testmodel = nanodeep.nanodeep_testmodel:main',
            'nanodeep_trainmodel = nanodeep.nanodeep_trainmodel:main',
            'draw_fast5_id = tool.draw_fast5_id:main'
          ]
      },

)
