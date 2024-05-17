
## **SANSpy**  
### Spanwise-averaged Navier-Stokes (SANS) equations modelling using a convolutional neural network

The SANS equations are used to reduce a turbulent flow presenting an homogeneous direction into a 2-D system, effectively cutting the computational cost of a simulation by orders of magnitude.
This is accomplished by including additional terms in the 2-D momentum equations which account for the 3-D turbulence mixing effects.
The additional unclosed terms are modelled here using a convolutional neural network (CNN).
Check [this preprint](https://arxiv.org/abs/2008.07528) for more info, or our journal publication:

> * Font, B., Weymouth, G.D., Nguyen, V.-T. & Tutty, O.R. (2021) Deep learning the spanwise-averaged Navier-Stokes equations. Journal of Computational Physics, 2021, 434(10):110199. [doi:10.1016/j.jcp.2021.110199](https://dx.doi.org/10.1016/j.jcp.2021.110199)

### Installation and usage

The use of a virtual environment is recommended for the installation of the package. Here we will use a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) environment based on Python 3.6. To create a new virtual environment: 
```
conda create --name my_env python=3.6
```
Now you can install this package into the `my_env` environment using `pip`:
```
git clone https://github.com/b-fg/sanspy
cd sanspy
source activate my_env
pip install -e ~/sanspy
```
The environment can be deactivated running `conda deactivate`. In order to be able to perform modification on the package without the need of reinstalling the `-e` (editable) argument is used. This provides the source path of the package to the `conda` environment so any modifications on the source code (the folder you have downloaded and installed from) is immediately available with no need to re-install.

To use the package just activate the `conda` environment in your terminal with `source activate my_env` or load it into your preferred IDE. In your Python script, you should now be able to import the modules with:

	from sanspy.model import Model
	
Scripts examples on training and testing are provided in the `examples/` folder.

	
### Dependencies
If there is a GPU in the current machine, this can be used to train or test the model (see in `examples/`).
In this sense, install the `keras-gpu` and package from `conda`, which will also install the `tensorflow` and `tensorflow-gpu` libraries as the backend:
```
conda install -c anaconda keras-gpu 
```

Finally, there is a dependency to my `postproc` package, which you can install from [this repository](https://github.com/b-fg/postproc).
This is used for reading the dataset, but you can just use your own read/write routines.

