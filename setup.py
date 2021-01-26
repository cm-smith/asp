from setuptools import setup

setup(
    name='asp'
    ,version='0.1'
    ,description='Assistant Stats Package'
    ,url='https://github.com/cm-smith/asp'
    ,author='Michael Smith'
    ,license='MIT'
    ,packages=['asp']
    ,test_suite='nose.collector'
    ,tests_require=['nose']
    ,install_requires=['numpy','pandas','statsmodels','scipy']
    ,python_requires='>=3.6'
    ,zip_safe=False
)
