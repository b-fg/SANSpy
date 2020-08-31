from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='sanspy',
    version='1.0',
    description='Spanwise-averaged Navier-Stokes equations modelling using a convolutional neural network',
    long_description=readme,
    author='Bernat Font Garcia',
    author_email='bernatfontgarcia@gmail.com',
    keywords=['turbulence, computational fluid dynamics, machine learning, neural networks'],
    url='https://github.com/b-fg/sanspy',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
		setup_requires=['numpy'],
		install_requires=['numpy', 'scipy', 'matplotlib', 'keras', 'tensorflow']
)
