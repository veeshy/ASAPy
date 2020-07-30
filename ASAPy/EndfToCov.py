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
import argparse
import pandas as pd

from ASAPy import AsapyCovStorage
from ASAPy import njoy
from ASAPy import CovManipulation
from ASAPy.data.data import ATOMIC_SYMBOL

_BOXER_TEMPLATE_COV = """0,{mat},{mt},{mat},{mt}
1,{mat},{mt},{mat},{mt}
2,{mat},{mt},{mat},{mt}
3,{mat},{mt},{mat},{mt}
0,0
"""

def _run_cover_chain(njoy_commands, tapein, tapeout, cover_tapes, mat_num, mts, input_filename=None, stdout=True,
                    njoy_exec='njoy', boxer_exec='boxer2mat', output_base_path='./',
                    use_temp_folder=True):
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
        List of MT numbers to get cov info for, use 1018 for chi
    input_filename : str, optional
        File name to write out NJOY input njoy_commands
    stdout : bool, optional
        Whether to display output when running NJOY
    njoy_exec : str, optional
        Path to NJOY executable
    base_file_name : str, optional
        The base file that the executable uses (likely tape or fort.)
    use_temp_folder : bool
        Option to output all intermediate files to a temp folder

    Raises
    ------
    subprocess.CalledProcessError
        If the executable process returns with a non-zero status

    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # override the tmpfile path
        if not use_temp_folder:
            tmpdir = output_base_path

        # Copy evaluations to appropriates 'tapes'
        for tape_num, filename in tapein.items():
            tmpfilename = os.path.join(tmpdir, 'tape{}'.format(tape_num))
            shutil.copy(filename, tmpfilename)

        run_program(njoy_commands, njoy_exec, stdout, tmpdir, os.path.join(output_base_path, input_filename))

        # Copy output files back to original directory
        shutil.copy(os.path.join(tmpdir, "output"), os.path.join(output_base_path, "njoy_output.txt"))

        for tape_num, filename in tapeout.items():
            tmpfilename = os.path.join(tmpdir, 'tape{}'.format(tape_num))
            if os.path.isfile(tmpfilename):
                shutil.copy(tmpfilename, os.path.join(output_base_path, filename))

        # Convert the cov boxer out to matrix form
        # each cover tape should be a different temp or material so loop over them and parse all mts
        for cover_tape in cover_tapes.items():
            for mt in mts:
                tape_num, file_name = cover_tape
                # the nin for boxer2matrix
                tempnjoycovtape = os.path.join(tmpdir, 'tape{}'.format(tape_num))
                tmpboxerfilename_in = os.path.join(tmpdir, 'fort.20')
                tmpboxerfilename_out = os.path.join(tmpdir, 'fort.21')
                shutil.copy(tempnjoycovtape, tmpboxerfilename_in)
                # use the same input multiple times for diff MTs
                # run boxer
                if mt == 1018:
                    boxer_commands = _BOXER_TEMPLATE_COV.format(mat=mat_num, mt=18)
                else:
                    boxer_commands = _BOXER_TEMPLATE_COV.format(mat=mat_num, mt=mt)

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
            correlation or cov matrix depending on njoy output

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

def run_cover_chain(endf_file, mts, temperatures, output_dir='./', cov_energy_groups=None,
                    iwt_fluxweight=9, user_flux_weight_vals=None, nu=False, chi=False, use_temp_folder=True,
                    njoy_exec='njoy',
                    boxer_exec='boxer2mat'):
    """
    Creates cov matrix and plots for mts and temperatures for the end file
    Parameters
    ----------
    endf_file
    mts
    temperatures
    output_dir
    cov_energy_groups
    iwt_fluxweight : int
        NJOY built-in flux weights: 3=1/E, 9=claw
    nu : bool
        Run with nubar cov
    chi : bool
        Run with chi cov
    use_temp_folder : bool
        Option to output all intermediate files to output_base_path
    """

    if not isinstance(mts, list):
        mts = [mts]

    mts = list(mts)

    if not isinstance(temperatures, list):
        temperatures = [temperatures]

    if not cov_energy_groups:
        raise Exception("Please specify energy group structure for cov groups")

    # ensure mt = 18 is processed, might not need this exception..
    if chi:
        if 18 not in mts:
            print("Automatically adding mt=18 needed for chi processing")
            mts.append(18)

    mat_num = njoy.get_mat_from_endf(endf_file)
    njoy_commands, tapein, tapeout = njoy.make_njoy_run(endf_file, temperatures=temperatures, pendf=None, error=0.001,
                                                        covr_plot_mts=mts, cov_energy_groups=cov_energy_groups,
                                                        broadr=True, heatr=False, purr=False, acer=False, errorr=True,
                                                        iwt_fluxweight=iwt_fluxweight,
                                                        user_flux_weight_vals=user_flux_weight_vals, nu=nu, chi=chi)

    # we know the order of tape out {covout1, plotout1,, covout2, plotout2, ...}, just get the cov outs
    cover_tapes = {item: value for item, value in list(tapeout.items())[0::2]}
    print(cover_tapes)

    if nu:
        mts.append(452)

    if chi:
        mts.append(1018)

    _run_cover_chain(njoy_commands, tapein, tapeout, cover_tapes, mat_num, mts, input_filename="testing_chain.txt",
                     stdout=True, njoy_exec=njoy_exec,
                     boxer_exec=boxer_exec, output_base_path=output_dir,
                     use_temp_folder=use_temp_folder)


def process_cov_to_h5(output_dir, zaid, mt, boxer_matrix_name='covr_300.txt_{mt}_matrix.txt',
                      output_h5_format='u235_102_{0}g_cov.h5'):
    rbo = read_boxer_out_matrix(os.path.join(output_dir, boxer_matrix_name.format(mt=mt)))
    group_bounds, xsec, std_dev, cov = rbo.get_block_data()
    groups = cov.shape[0]
    output_h5_name = output_h5_format.format(groups)

    # covert the cov read to corr
    corr = CovManipulation.cov_to_correlation(cov)

    with pd.HDFStore(os.path.join(output_dir, output_h5_name), 'a') as h:
        df = AsapyCovStorage.create_corr_df(groups)

        # make sure diag is all ones
        np.fill_diagonal(corr, 1.0)
        df.loc[:, :] = corr

        # if std_dev was zero anywhere,
        if np.any(std_dev == 0):
            print("Found 0 std_dev for {0} xsec, correcting corr to not have any NaNs.".format(sum(std_dev == 0)))
            df.loc[std_dev == 0, :] = 0
            df.loc[:, std_dev == 0] = 0

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
        # set all NaN's (which presumably had 0 xsec) to sensible values for future sampling
        # values set to 1.0 \pm 1e-15 so that the point can be sampled and it won't change off of 1.0
        df.loc[df['rel.s.d.(1)'].isna(), ('x-sec(1)', 'x-sec(2)')] = 1.0
        df.loc[df['rel.s.d.(1)'].isna(), ('s.d.(1)', 's.d(2)')] = 1e-7
        df.loc[df['rel.s.d.(1)'].isna(), ('rel.s.d.(1)', 'rel.s.d(2)')] = 1e-7

        AsapyCovStorage.add_stddev_to_store(h, df, zaid, mt, zaid, mt)

def parse_args(args):
    """
    Create CLI parser and does any data formatting

    Parameters
    ----------
    args : list
        CLI sys args

    Returns
    -------
    argparse
        Parsed arguments (access via dot operator)

    """

    parser = argparse.ArgumentParser(
        description="Generate table-wise covariance data from ENDF data using NJOY", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('endf_file', help="The base ENDF file to extract covariance data from")

    parser.add_argument('-energy_bin_structure',
                        choices=["SCALE_3", "SCALE_44", "SCALE_56", "SCALE_238", "SCALE_252"],
                        help="The energy bin structure", default='SCALE_252')

    parser.add_argument("-output_path", help="Path to where all outputs should be placed.", default=None)

    parser.add_argument('-output_store', help="""The HDF5 store name to add covariance data to. Defaults to ZAID_#groups.
Data will be overridden. Only store one energy_bin_structure per store
with any number of nuclides and MTs. _#groups is automatically
appended to the given name""", default=None)
    parser.add_argument('-njoy_spectrum_weighting', default=6, type=int, choices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        help="""The NJOY spectrum weight magic number:
iwt          meaning
---          -------
 1           read in smooth weight function
 2           constant
 3           1/e
 4           1/e + fission spectrum + thermal maxwellian
 5           epri-cell lwr
 6           (thermal) -- (1/e) -- (fission + fusion)
 7           same with t-dep thermal part
 8           thermal--1/e--fast reactor--fission + fusion
 9           claw weight function
10           claw with t-dependent thermal part
11           vitamin-e weight function (ornl-5505)
12           vit-e with t-dep thermal part
""")

    parser.add_argument('-mts', help="The reaction MT numbers to sample, defaults to all available", type=int, nargs='+')
    parser.add_argument('-temperature',
                        help="The temperature to broaden the cross-section to for covariance representation",
                        type=int, default=300)
    parser.add_argument('-njoy_exec', help="Path to your NJOY executable if it's not already in your $PATH",
                        default='njoy')
    parser.add_argument('-boxer_exec', help="Path to your boxer executable if it's not already in your $PATH",
                        default='boxer2mat')

    # maybe add this later
    # parser.add_argument('--writepbs', action='store_true',
    #                     help="Creates a pbs file to run this function (requires mcACE)")
    # parser.add_argument('--waitforjob', help="Job number to wait for until this job runs (requires mcACE)")
    # parser.add_argument('--subpbs', action='store_true', help="Runs the created pbs file (requires mcACE)")


    parsed_args = parser.parse_args(args)

    # ensure full path of the input / output is given to make sure we can execute the input from any
    # location and output to any location without knowing where python was invoked
    parsed_args.endf_file = os.path.abspath(parsed_args.endf_file)
    if parsed_args.output_path:
        parsed_args.output_path = os.path.abspath(parsed_args.output_path)
    else:
        parsed_args.output_path = './'

    return parsed_args

if __name__ == "__main__":
    import sys
    import warnings
    import tables

    # hide the warning where naming the HDF5 storage key starting with a number is not considered a
    # natural name in tables. HDF5 doesn't care for pandas access so it's okay.
    warnings.simplefilter("ignore", tables.NaturalNameWarning)

    args = parse_args(sys.argv[1:])

    output_dir = args.output_path
    endf_file = args.endf_file

    ev = njoy.endf.Evaluation(endf_file)
    zaid = int("{0}{1}".format(ev.target['atomic_number'], str(ev.target['mass_number']).zfill(3)))

    # get all cov data on this endf file
    available_mts = []
    for r in ev.reaction_list:
        # r contains the mf # then mt #, mf33 is cov data
        if r[0] == 33:
            available_mts.append(r[1])

    # make sure some cov data is available
    if len(available_mts) == 0:
        raise Exception(f"No covariance data found in {args.endf_file}")

    # if user asked for specific MTs, make sure they are available. If so use them else use all available
    if args.mts:
        if not set(args.mts).issubset(set(available_mts)):
            raise Exception(f"One of the specified covariance MTs ({args.mts}) was not found in the ENDF file (found {available_mts})")

        mts = args.mts
    else:
        mts = available_mts

    # output as Z# followed by A# followed by _ # of groups
    if args.output_store is None:
        output_h5_format = f"{ATOMIC_SYMBOL[ev.target['atomic_number']]}{ev.target['mass_number']}_{{0}}g.h5"
    else:
        output_h5_format = args.output_store + "{0}g.h5"

    # need to do nu-bar and chi cov getting if fissionable.
    if ev.target['fissionable'] == True:
        nu = True
        chi = True
    else:
        nu = False
        chi = False

    # user selection desired here
    if args.energy_bin_structure == "SCALE_3":
        cov_groups = njoy.energy_groups_3
    elif args.energy_bin_structure == "SCALE_44":
        cov_groups = njoy.energy_groups_44
    elif args.energy_bin_structure == "SCALE_56":
        cov_groups = njoy.energy_groups_56
    elif args.energy_bin_structure == "SCALE_238":
        cov_groups = njoy.energy_groups_238
    elif args.energy_bin_structure == "SCALE_252":
        cov_groups = njoy.energy_groups_252

    # from low to high, (e, flux_val) pairs
    # one day we can expose this to the user
    user_flux_weight_vals = None

    # T should be user input, for each T?
    temperature = args.temperature
    run_cover_chain(endf_file, mts, [temperature],
                    output_dir=output_dir, cov_energy_groups=cov_groups, iwt_fluxweight=args.njoy_spectrum_weighting,
                    user_flux_weight_vals=user_flux_weight_vals, nu=nu, chi=chi, use_temp_folder=False,
                    njoy_exec=args.njoy_exec,
                    boxer_exec=args.boxer_exec)
    for mt in mts:
        process_cov_to_h5(output_dir, zaid, mt, boxer_matrix_name=f'covr_{temperature}.txt_{{mt}}_matrix.txt',
                          output_h5_format=output_h5_format)

    if nu:
        process_cov_to_h5(output_dir, zaid, 452, boxer_matrix_name=f'covr_nu_{temperature}.txt_{{mt}}_matrix.txt',
                          output_h5_format=output_h5_format)

    if chi:
        process_cov_to_h5(output_dir, zaid, 1018, boxer_matrix_name=f'covr_chi_{temperature}.txt_{{mt}}_matrix.txt',
                          output_h5_format=output_h5_format)
