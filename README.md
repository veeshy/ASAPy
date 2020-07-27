# ASAPY

A python based method to sample ACE data based on ENDF covariances.

See ASAPY-dissertation_section.pdf for more background.

Recommended install via 

```
python -m venv env
. env/bin/activate
pip install .
pytest
```

`mpi4py` is a requirement that means you need `mpicc` on your system. You can get it easily with conda https://anaconda.org/conda-forge/mpich-mpicc if you don't mind mixing conda/pip and what not. You can build it yourself too.
