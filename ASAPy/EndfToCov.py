"""
Takes in an ENDF file that has a covariance on it then creates NJOY
inputs to generate group covariances in BOXER format. This format is
converted to matrix format then parsed into a ASAPy covariance store.
The result can be appended to an existing store or made into a new store.
"""

import os
import shutil
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import tempfile

import numpy as np

from ASAPy import AsapyCovStorage
from ASAPy import njoy

_BOXER_TEMPLATE = """0,{mat},{mt},{mat},{mt}
1,{mat},{mt},{mat},{mt}
2,{mat},{mt},{mat},{mt}
4,{mat},{mt},{mat},{mt}
0,0
"""

def _run_cover_chain(njoy_commands, tapein, tapeout, cover_tapes, mat_num, mts, input_filename=None, stdout=True,
                    njoy_exec='../boxer2mat/boxer2mat', boxer_exec='../boxer2mat/boxer2mat', output_base_path='./'):
    """Run NJOY to create a cov matrix in easily readable format by
    converting the BOXER output to matrix form

    Parameters
    ----------
    njoy_commands : str
        Input njoy_commands for NJOY
    tapein : dict
        Dictionary mapping tape numbers to paths for any input files
    tapeout : dict
        Dictionary mapping tape numbers to paths for any output files
    mts : list
        List of MT numbers to get cov info for
    input_filename : str, optional
        File name to write out NJOY input njoy_commands
    stdout : bool, optional
        Whether to display output when running NJOY
    njoy_exec : str, optional
        Path to NJOY executable
    base_file_name : str, optional
        The base file that the executable uses (likely tape or fort.)

    Raises
    ------
    subprocess.CalledProcessError
        If the executable process returns with a non-zero status

    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy evaluations to appropriates 'tapes'
        for tape_num, filename in tapein.items():
            tmpfilename = os.path.join(tmpdir, 'tape{}'.format(tape_num))
            shutil.copy(filename, tmpfilename)

        run_program(njoy_commands, njoy_exec, stdout, tmpdir, os.path.join(output_base_path, input_filename))

        # Copy output files back to original directory
        for tape_num, filename in tapeout.items():
            tmpfilename = os.path.join(tmpdir, 'tape{}'.format(tape_num))
            if os.path.isfile(tmpfilename):
                shutil.copy(tmpfilename, os.path.join(output_base_path, filename))

        # Convert the cov boxer out to matrix form
        # for loop since several temps might be evaluated
        for tape_num, file_name in cover_tapes.items():
            # the nin for boxer2matrix
            tempnjoycovtape = os.path.join(tmpdir, 'tape{}'.format(tape_num))
            tmpboxerfilename_in = os.path.join(tmpdir, 'fort.20')
            tmpboxerfilename_out = os.path.join(tmpdir, 'fort.21')
            shutil.move(tempnjoycovtape, tmpboxerfilename_in)
            # use the same input multiple times for diff MTs
            for mt in mts:
                # run boxer
                boxer_commands = _BOXER_TEMPLATE.format(mat=mat_num, mt=mt)
                run_program(boxer_commands, boxer_exec, stdout, tmpdir, os.path.join(output_base_path, input_filename + "boxer.txt"))
                # save the output locally
                shutil.move(tmpboxerfilename_out, os.path.join(output_base_path, file_name + "_" + str(mt) + "_matrix.txt"))


def run_program(commands, executable_path, stdout, run_dir, write_input_file=None):
    """
    Run
    Parameters
    ----------
    commands : str
        Input commands for executable
    executable_path : str
        Path to executable
    stdout : bool, optional
        Whether to display output when running NJOY
    run_dir : str
        Directory to run in
    write_input_file : str, optional
        File name to write out input commands

    Raises
    ------
    subprocess.CalledProcessError
        If the executable process returns with a non-zero status

    """

    if write_input_file is not None:
        with open(write_input_file, 'a') as f:
            f.write(commands)
            f.write("\n")

    # Start up process
    executable = Popen([executable_path], cwd=run_dir, stdin=PIPE, stdout=PIPE,
                 stderr=STDOUT, universal_newlines=True)
    executable.stdin.write(commands)
    executable.stdin.flush()
    lines = []

    while True:
        # If process is finished, break loop
        line = executable.stdout.readline()
        if not line and executable.poll() is not None:
            break

        lines.append(line)
        if stdout:
            # If user requested output, print to screen
            print(line, end='')
    # Check for error
    if executable.returncode != 0:
        raise CalledProcessError(executable.returncode, executable_path,
                                 ''.join(lines))

    executable.stdin.close()
    if stdout:
        executable.stdout.close()


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
        rel_dev = self._block_lines_to_array(self.block_line_nums[2] + 1, self.block_line_nums[3])
        std_dev = rel_dev * xsec
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

def run_cover_chain(endf_file, mts, temperatures, output_dir='./', cov_energy_groups=None):
    """
    Creates cov matrix and plots for mts and temperatures for the end file
    Parameters
    ----------
    endf_file
    mts
    temperatures

    Returns
    -------

    """

    if not isinstance(mts, list):
        mts = [mts]

    if not isinstance(temperatures, list):
        temperatures = [temperatures]

    if not cov_energy_groups:
        cov_energy_groups = njoy.energy_groups_238

    mat_num = njoy.get_mat_from_endf(endf_file)
    njoy_commands, tapein, tapeout = njoy.make_njoy_run(endf_file, temperatures=temperatures, pendf=None, error=0.001,
                                                        covr_plot_mts=mts, cov_energy_groups=cov_energy_groups,
                                                        broadr=True, heatr=False, purr=False, acer=False, errorr=True)

    # we know the order of tape out {covout1, plotout1,, covout2, plotout2, ...}, just get the cov outs
    cover_tapes = {item: value for item, value in list(tapeout.items())[0::2]}

    _run_cover_chain(njoy_commands, tapein, tapeout, cover_tapes, mat_num, mts, input_filename="testing_chain.txt",
                     stdout=True, njoy_exec='/Users/veeshy/projects/NJOY2016/bin/njoy',
                     boxer_exec='/Users/veeshy/projects/ASAPy/boxer2mat/boxer2mat', output_base_path=output_dir)

if __name__ == "__main__":
    import pandas as pd

    # run_cover_chain("../test_data/n_0125_1-H-1.dat", [18], [300],
    #                 output_dir='../run_cover_chain_test_out/',
    #                 cov_energy_groups=njoy.energy_groups_44)

    # rbo = read_boxer_out_matrix('../run_cover_chain_test_out/covr_2400.txt_2_matrix.txt')
    #
    # group_bounds, xsec, std_dev, cov = rbo.get_block_data()
    #
    # groups = 238
    #
    # with pd.HDFStore('test_put.h5', 'w') as h:
    #
    #     df = AsapyCovStorage.create_corr_df(groups)
    #
    #     df.loc[:, :] = cov
    #     AsapyCovStorage.add_corr_to_store(h, df, '1001', '102', '1001', '102')
    #     df = AsapyCovStorage.create_stddev_df(groups)
    #
    #     df['e high'] = group_bounds[0:-1]
    #     df['e low'] = group_bounds[1:]
    #     df['x-sec(1)'] = xsec
    #     df['x-sec(2)'] = xsec
    #     df['rel.s.d.(1)'] = std_dev / xsec
    #     df['rel.s.d(2)'] = std_dev / xsec
    #     df['s.d.(1)'] = std_dev
    #     df['s.d(2)'] = std_dev
    #
    #     AsapyCovStorage.add_stddev_to_store(h, df, '1001', '102', '1001', '102')

    mt = 102
    run_cover_chain("/Users/veeshy/Downloads/ENDF-B-VII.1/neutrons/n-092_U_238.endf", [mt], [300],
                    output_dir='../u238_102_3_group/',
                    cov_energy_groups=njoy.energy_groups_3)

    rbo = read_boxer_out_matrix('../u238_102_3_group/covr_300.txt_{mt}_matrix.txt'.format(mt=mt))

    group_bounds, xsec, std_dev, cov = rbo.get_block_data()

    groups = cov.shape[0]

    with pd.HDFStore('test_put.h5', 'a') as h:

        zaid = 92238


        df = AsapyCovStorage.create_corr_df(groups)

        df.loc[:, :] = cov
        AsapyCovStorage.add_corr_to_store(h, df, zaid, mt, zaid, mt)
        df = AsapyCovStorage.create_stddev_df(groups)

        df['e high'] = group_bounds[0:-1]
        df['e low'] = group_bounds[1:]
        df['x-sec(1)'] = xsec
        df['x-sec(2)'] = xsec
        df['rel.s.d.(1)'] = std_dev / xsec
        df['rel.s.d(2)'] = std_dev / xsec
        df['s.d.(1)'] = std_dev
        df['s.d(2)'] = std_dev

        AsapyCovStorage.add_stddev_to_store(h, df, zaid, mt, zaid, mt)


# for each mat/mt cov:
#   read the tape.21 outputs (mat_mt_mat_mt.mat)
#   parse the group bounds, xsec, std_dev, correlation matrix  done!
#   create stddev and corr df
#   store these