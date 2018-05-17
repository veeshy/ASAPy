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

import numpy as np

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


#
# njoy.make_njoy_run('../data/e8/tape20', temperatures=[300],
#                    broadr=True, heatr=False, purr=False, acer=False, errorr=True,
#                    cov_energy_groups=njoy.energy_groups_44,
#                    **{'input_filename': '../data/cov_u235.i', 'stdout': True})

class read_boxer_out_matrix:
    """
    Reads file, the output of boxer2matrix. File format is such that 3 blocks of data exist

    Parameters
    ----------
    file : str
        Path to boxer2matrix file
    """

    def __init__(self, file):

        with open(file, 'r') as f:
            self.lines = f.readlines()

        self.block_line_nums = self._find_block_line_nums()

    def _find_block_line_nums(self):
        """
        Finds the block line #s where data begins

        1 for the group bounds, another for the xsec, and a third for the cov.

        0 more header info for groups
        lines of data

        1 more header info for xsec
        lines of data

        2 more header info for std dev
        lines of data

        3 more header info for cov
        lines of data

        Returns
        -------
        list
            Line numbers
        """
        # find the line start numbers of the data blocks
        block_start_line_nums = []
        for line_num, line in enumerate(self.lines):
            l = line.split()
            if l:
                # the only ints as the first block of char are ints, so if we find an int we know we have a start of a block
                try:
                    int(l[0])
                    block_start_line_nums.append(line_num)
                except ValueError:
                    pass

        if not block_start_line_nums:
            raise Exception("Did not find any block line numbers.")

        return block_start_line_nums

    def get_block_data(self):
        """

        Returns
        -------
        np.array
            group bounds (eV)
        np.array
            xsec
        np.array
            std dev
        np.array
            correlation matrix

        """

        group_bounds = self._block_lines_to_array(self.block_line_nums[0] + 1, self.block_line_nums[1])
        xsec = self._block_lines_to_array(self.block_line_nums[1] + 1, self.block_line_nums[2])
        std_dev = self._block_lines_to_array(self.block_line_nums[2] + 1, self.block_line_nums[3])
        cov = self._block_lines_to_array(self.block_line_nums[3] + 1, len(self.lines))

        number_of_xsec = len(xsec)
        cov = np.reshape(cov, [number_of_xsec, number_of_xsec])

        return group_bounds, xsec, std_dev, cov

    def _block_lines_to_array(self, line_num_start, line_num_end):
        """
        Converts a block of lines into an np.array
        Parameters
        ----------
        line_num_start : the first line to read
        line_num_end : the last line to read (this line # is the end point of the index so it is NOT read

        Returns
        -------
        np.array

        """
        block_lines = self.lines[line_num_start:line_num_end]
        block_lines_no_newlines = []

        # need to sanitize this against very long values since this is a fixed len format
        for line in block_lines:
            replace_white_space_with_zero = line.replace(' ', '0')
            removed_new_lines = ''.join(replace_white_space_with_zero.split())
            block_lines_no_newlines.append(removed_new_lines)

        block_lines_str = ''.join(block_lines_no_newlines)

        string_width = 10
        # split on every 10 chars
        block_lines_str = [block_lines_str[i:i + string_width] for i in range(0, len(block_lines_str), string_width)]
        # rejoin string, leaving a space so that numpy can read it nicely
        block_lines_str = ' '.join(block_lines_str)

        values = np.fromstring(block_lines_str, sep=' ')

        return values


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