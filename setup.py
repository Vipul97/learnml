import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='learnml',
    version='0.2',
    author='Vipul Gharde',
    description='LearnML is a Python module for Machine Learning and Deep Learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Vipul97/learnml',
    packages=setuptools.find_packages(),
    install_requires=['graphviz', 'matplotlib', 'numpy', 'pandas'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    python_requires='>=3.5'
)
