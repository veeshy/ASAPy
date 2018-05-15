"""
Takes in an ENDF file that has a covariance on it then creates NJOY
inputs to generate group covariances in BOXER format. This format is
converted to matrix format then parsed into a ASAPy covariance store.
The result can be appended to an existing store or made into a new store.
"""

from collections import namedtuple
from io import StringIO
import os
import shutil
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import sys
import tempfile

from ASAPy import AsapyCovStorage
from ASAPy import njoy


def run(commands, tapein, tapeout, input_filename=None, stdout=False,
        njoy_exec='../boxer2mat/boxer2mat'):
    """Run NJOY with given commands

    Parameters
    ----------
    commands : str
        Input commands for NJOY
    tapein : dict
        Dictionary mapping tape numbers to paths for any input files
    tapeout : dict
        Dictionary mapping tape numbers to paths for any output files
    input_filename : str, optional
        File name to write out NJOY input commands
    stdout : bool, optional
        Whether to display output when running NJOY
    njoy_exec : str, optional
        Path to NJOY executable

    Raises
    ------
    subprocess.CalledProcessError
        If the NJOY process returns with a non-zero status

    """

    if input_filename is not None:
        with open(input_filename, 'w') as f:
            f.write(commands)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy evaluations to appropriates 'tapes'
        for tape_num, filename in tapein.items():
            tmpfilename = os.path.join(tmpdir, 'tape{}'.format(tape_num))
            shutil.copy(filename, tmpfilename)

        # Start up NJOY process
        njoy = Popen([njoy_exec], cwd=tmpdir, stdin=PIPE, stdout=PIPE,
                     stderr=STDOUT, universal_newlines=True)

        njoy.stdin.write(commands)
        njoy.stdin.flush()
        lines = []
        while True:
            # If process is finished, break loop
            line = njoy.stdout.readline()
            if not line and njoy.poll() is not None:
                break

            lines.append(line)
            if stdout:
                # If user requested output, print to screen
                print(line, end='')

        # Check for error
        if njoy.returncode != 0:
            raise CalledProcessError(njoy.returncode, njoy_exec,
                                     ''.join(lines))

        # Copy output files back to original directory
        for tape_num, filename in tapeout.items():
            tmpfilename = os.path.join(tmpdir, 'tape{}'.format(tape_num))
            if os.path.isfile(tmpfilename):
                shutil.move(tmpfilename, filename)



njoy.make_njoy_run('../data/e8/tape20', temperatures=[300],
                   broadr=True, heatr=False, purr=False, acer=False, errorr=True,
                   cov_energy_groups=njoy.energy_groups_44,
                   **{'input_filename': '../data/cov_u235.i', 'stdout': True})

# need to:
#  copy tape24 (the njoy cover output) to fort.20 (the boxer input)
#  make sure we know the NJOY mat # (endf has that, in njoy.py I think)
#  for each mat/mt cov we want:
#   create an input that makes the group bounds, xsec, std_dev, and correlation
#   run boxer2mat
#   mv the output (tape.21) to mat_mt_mat_mt.mat

# for each mat/mt cov:
#   read the tape.21 outputs (mat_mt_mat_mt.mat)
#   parse the group bounds, xsec, std_dev, correlation matrix
#   create stddev and corr df
#   store these