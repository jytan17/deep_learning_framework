import setuptools

setuptools.setup(
    name='pydeep',
    version='0.0.1',
    author='Junyong Tan',
    author_email='jtan9801@gmail.com',
    description='A small deep learning package',
    url='https://github.com/jytan17/deep_learning_framework',
    project_urls = {
        "Bug Tracker": "https://github.com/jytan17/deep_learning_framework/issues"
    },
    license='MIT',
    packages=['pydeep'],
    install_requires=['numpy'],
)
