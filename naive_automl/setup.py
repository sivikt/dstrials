from setuptools import find_packages, setup


setup(
    name='naive_automl',
    version='0.1.0',
    description='AutoML Regularized Logistic Regression',
    author='Sergei Sintsov',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    python_requires='~=3.7',
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy~=1.18.1',
        'pandas~=1.0.1',
        'scikit-learn~=0.22.1',
        'scipy~=1.3.1'
    ]
)
