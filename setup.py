import setuptools

setuptools.setup(
    name='pydeep',
    version='0.0.1',
    author='Junyong Tan',
    author_email='jtan9801@gmail.com',
    description='A small machine learning package',
    url='https://github.com/jytan17/deeplearning_framework',
    project_urls = {
        "Bug Tracker": "https://github.com/jytan17/deeplearning_framework/issues"
    },
    license='MIT',
    packages=['models'],
    install_requires=['numpy'],
)
