from setuptools import setup, find_packages

setup(
    name='gymnasium_connect_four',
    version='1.3.5',
    description='A connect 4 (connect four) environment for OpenAI Gym and Gymnasium',
    author='Lucas Bertola',
    url='https://github.com/lucasBertola/Connect-4-env',  
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
        'pygame',
    ]
)
