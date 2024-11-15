import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='learnml',
    version='0.11.1',
    author='Vipul Gharde',
    authon_email='vipul.gharde@gmail.com',
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
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    python_requires='>=3.11'
)
