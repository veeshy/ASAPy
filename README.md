# ASAPY

A python based method to sample ACE data based on ENDF covariances.

See ASAPY-dissertation_section.pdf for more background.

Recommended install via 

```commandline
python -m venv env
. env/bin/activate
pip install .
# these probably all won't pass because of some hard coded njoy/boxer2mat paths, sorry!
pytest
```

`mpi4py` is a requirement that means you need `mpicc` on your system. You can get it easily with conda https://anaconda.org/conda-forge/mpich-mpicc if you don't mind mixing conda/pip and what not. You can build it yourself too.

`NJOY` is a requirement which you can get from https://github.com/njoy/NJOY2016 which also has installation instructions

`boxer2mat` is included in this repo which was copied from the NJOY manual because it is not distributed with NJOY. You can cd into that folder and simply type `make` if you have gfortran. If not you can easily edit the Makefile to use the fortran compiler of your choice (no really, the make file has 4 lines in it)

You'd then supply ENDF files to `ENDFToCov.py` which will extract covariance data from ENDF data (not included here) and make HDF5 stores of the data.

Then you can sample ACE files (not included here) from those covariances using various distributions with `XsecSampler.py`.

Shout out to openMC for their ACE data reader, included in ASAPy/data with their license requirement.

An example use case would be to process an ENDF file to get all the covariance matrices stored on the file by:

```commandline
python ./ASAPy/EndfToCov.py ENDF_FILE -energy_bin_structure SCALE_252 -boxer_exec /Users/veeshy/projects/ASAPy/boxer2mat/boxer2mat -njoy_exec /Users/veeshy/projects/NJOY2016/bin/njoy
```  

That will result in an HDF store with multi-group cross-sections, std-deviations, and covariance matrices.

You can then use this information to sample ACE data files via below which would draw 500 samples varying only mt=102 via lognormal sampling.

```commandline
python ./ASAPy/XsecSampler.py ACE_FILE HDF_FILE_CREATED_IN_PREVIOUS_STEP 500 102 --make_plots -distribution lognormal
```