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
from ASAPy import CovManipulation

_BOXER_TEMPLATE = """0,{mat},{mt},{mat},{mt}
1,{mat},{mt},{mat},{mt}
2,{mat},{mt},{mat},{mt}
4,{mat},{mt},{mat},{mt}
0,0
"""

_BOXER_TEMPLATE_COV = """0,{mat},{mt},{mat},{mt}
1,{mat},{mt},{mat},{mt}
2,{mat},{mt},{mat},{mt}
3,{mat},{mt},{mat},{mt}
0,0
"""

def _run_cover_chain(njoy_commands, tapein, tapeout, cover_tapes, mat_num, mts, input_filename=None, stdout=True,
                    njoy_exec='../boxer2mat/boxer2mat', boxer_exec='../boxer2mat/boxer2mat', output_base_path='./',
                    use_temp_folder=True, chi=False):
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
    use_temp_folder : bool
        Option to output all intermediate files to a temp folder
    chi : bool
        Use when processing fission chi (mt=18, mf=5) because we use cov for this due to cover output issues

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
                if mt == 18 and chi:
                    boxer_commands = _BOXER_TEMPLATE_COV.format(mat=mat_num, mt=mt)
                else:
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

def run_cover_chain(endf_file, mts, temperatures, output_dir='./', cov_energy_groups=None,
                    iwt_fluxweight=9, user_flux_weight_vals=None, nu=False, chi=False, use_temp_folder=True):
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
        cov_energy_groups = njoy.energy_groups_238

    # ensure mt = 18 is processed, might not need this exception..
    if chi:
        if 18 in mts:
            raise Exception("Ambiguous parsing when both fission chi spectrum and mt=18 requested, please do one at a time.")
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

    _run_cover_chain(njoy_commands, tapein, tapeout, cover_tapes, mat_num, mts, input_filename="testing_chain.txt",
                     stdout=True, njoy_exec='/Users/veeshy/projects/NJOY2016/bin/njoy',
                     boxer_exec='/Users/veeshy/projects/ASAPy/boxer2mat/boxer2mat', output_base_path=output_dir,
                     use_temp_folder=use_temp_folder, chi=chi)


def process_cov_to_h5(output_dir, zaid, mt, boxer_matrix_name='covr_300.txt_{mt}_matrix.txt',
                      output_h5_format='u235_102_{0}g_cov.h5', cov_in_boxer=False):
    rbo = read_boxer_out_matrix(output_dir + boxer_matrix_name.format(mt=mt))
    group_bounds, xsec, std_dev, corr = rbo.get_block_data()
    groups = corr.shape[0]
    output_h5_name = output_h5_format.format(groups)

    if cov_in_boxer:
        # covert the cov read to corr
        corr = CovManipulation.cov_to_correlation(corr)

    with pd.HDFStore(output_dir + output_h5_name, 'a') as h:
        df = AsapyCovStorage.create_corr_df(groups)

        # make sure diag is all ones
        np.fill_diagonal(corr, 1.0)
        df.loc[:, :] = corr
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

    mts = [18, 102]
    mts = []
    nu = False
    chi = True
    zaid = 92235
    output_dir = '/Users/veeshy/projects/ASAPy/u235_viii/'
    # endf_file = "/Users/veeshy/Downloads/ENDF-B-VII.1/neutrons/n-092_U_235.endf"
    endf_file = "/Users/veeshy/Downloads/ENDF-B-VIII.0_neutrons/n-092_U_235.endf"

    # cov_groups = [njoy.energy_groups_44, njoy.energy_groups_56, njoy.energy_groups_252]
    cov_groups = [njoy.energy_groups_56]

    # from low to high, (e, flux_val) pairs
    user_flux_weight_vals = [1e-05, 0, 0.0001, 0, 0.0005, 0, 0.00075, 0, 0.001, 0, 0.0012, 0, 0.0015, 0, 0.002, 0,
                             0.0025, 0, 0.003, 0, 0.004, 0, 0.005, 0, 0.0075, 0, 0.01, 0, 0.0253, 0, 0.03, 0, 0.04, 0,
                             0.05, 0, 0.06, 0, 0.07, 0, 0.08, 0, 0.09, 0, 0.1, 0, 0.125, 0, 0.15, 0, 0.175, 0, 0.2, 0,
                             0.225, 0, 0.25, 0, 0.275, 0, 0.3, 1.65542e-16, 0.325, 0, 0.35, 0, 0.375, 0, 0.4, 0, 0.45,
                             2.03688e-14, 0.5, 0, 0.55, 0, 0.6, 0, 0.625, 0, 0.65, 0, 0.7, 4.32499e-14, 0.75, 0, 0.8, 0,
                             0.85, 0, 0.9, 0, 0.925, 0, 0.95, 0, 0.975, 0, 1, 0, 1.01, 0, 1.02, 0, 1.03, 0, 1.04, 0,
                             1.05, 0, 1.06, 0, 1.07, 0, 1.08, 0, 1.09, 0, 1.1, 0, 1.11, 0, 1.12, 0, 1.13, 0, 1.14, 0,
                             1.15, 0, 1.175, 0, 1.2, 0, 1.225, 0, 1.25, 0, 1.3, 0, 1.35, 0, 1.4, 0, 1.45, 0, 1.5, 0,
                             1.59, 7.2581e-15, 1.68, 0, 1.77, 3.730415e-13, 1.86, 6.023505e-12, 1.94, 3.867265e-11, 2,
                             6.094728e-14, 2.12, 8.70778e-15, 2.21, 1.24952e-14, 2.3, 2.766103e-14, 2.38, 1.68347e-14,
                             2.47, 2.942096e-14, 2.57, 3.167368e-13, 2.67, 2.956726e-14, 2.77, 0, 2.87, 0, 2.97,
                             1.20836e-14, 3, 0, 3.05, 0, 3.15, 0, 3.5, 1.176057e-13, 3.73, 3.062711e-13, 4,
                             1.890825e-13, 4.75, 6.664941e-12, 5, 4.179762e-13, 5.4, 1.698593e-12, 6, 4.484307e-12,
                             6.25, 5.894356e-13, 6.5, 1.605892e-13, 6.75, 2.156568e-13, 7, 1.402583e-12, 7.15,
                             7.535842e-14, 8.1, 7.976481e-12, 9.1, 1.0098e-12, 10, 1.026129e-12, 11.5, 5.252656e-12,
                             11.9, 3.260186e-13, 12.9, 4.133419e-12, 13.75, 6.776711e-12, 14.4, 1.507322e-12, 15.1,
                             7.867332e-12, 16, 1.189735e-11, 17, 9.150512e-12, 18.5, 1.757143e-11, 19, 2.747768e-12, 20,
                             3.67381e-12, 21, 3.888112e-12, 22.5, 2.08702e-11, 25, 1.298812e-11, 27.5, 2.248686e-11, 30,
                             4.320085e-11, 31.25, 1.24734e-11, 31.75, 6.86865e-12, 33.25, 3.885563e-11, 33.75,
                             1.001918e-12, 34.6, 2.85754e-12, 35.5, 8.181113e-13, 37, 1.083295e-11, 38, 1.783616e-11,
                             39.1, 2.413766e-11, 39.6, 6.692044e-13, 41, 8.652972e-12, 42.4, 2.003533e-11, 44,
                             1.016158e-11, 45.2, 4.679973e-12, 47, 1.799009e-11, 48.3, 9.91449e-12, 49.2, 3.719541e-12,
                             50.6, 7.328506e-12, 52, 4.051688e-12, 53.4, 9.098653e-12, 59, 2.462898e-11, 61,
                             1.406477e-11, 65, 5.262538e-11, 67.5, 4.677212e-11, 72, 6.570865e-11, 76, 3.826738e-11, 80,
                             6.479184e-11, 82, 2.153785e-11, 90, 1.12918e-10, 100, 1.760217e-10, 108, 1.979459e-10, 115,
                             2.685163e-10, 119, 3.367076e-11, 122, 7.395871e-11, 186, 1.451303e-09, 192.5, 1.58983e-10,
                             207.5, 3.132341e-10, 210, 4.037133e-11, 240, 4.729311e-10, 285, 8.961192e-10, 305,
                             4.721722e-10, 550, 5.977819e-09, 670, 4.390113e-09, 683, 3.272614e-10, 950, 1.058667e-08,
                             1150, 7.984139e-09, 1500, 1.666035e-08, 1550, 3.127989e-09, 1800, 1.497921e-08, 2200,
                             2.438634e-08, 2290, 7.199611e-09, 2580, 2.452208e-08, 3000, 4.516569e-08, 3740,
                             8.00608e-08, 3900, 1.755489e-08, 6000, 2.965444e-07, 8030, 4.16639e-07, 9500, 3.652173e-07,
                             13000, 1.076901e-06, 17000, 1.598843e-06, 25000, 4.040156e-06, 30000, 3.004691e-06, 45000,
                             1.124604e-05, 50000, 4.44385e-06, 52000, 1.852986e-06, 60000, 7.855586e-06, 73000,
                             1.450502e-05, 75000, 2.404877e-06, 82000, 8.765053e-06, 85000, 3.92272e-06, 100000,
                             2.047594e-05, 128300, 4.226679e-05, 150000, 3.486128e-05, 200000, 8.41866e-05, 270000,
                             0.0001190256, 330000, 9.877645e-05, 400000, 0.0001112772, 420000, 3.06855e-05, 440000,
                             3.015731e-05, 470000, 4.399281e-05, 499520, 4.228211e-05, 550000, 6.942029e-05, 573000,
                             3.041276e-05, 600000, 3.473289e-05, 670000, 8.478525e-05, 679000, 1.033917e-05, 750000,
                             7.777236e-05, 820000, 7.075093e-05, 861100, 3.917491e-05, 875000, 1.283088e-05, 900000,
                             2.259757e-05, 920000, 1.759737e-05, 1010000, 7.465441e-05, 1100000, 6.84441e-05, 1200000,
                             6.942249e-05, 1250000, 3.257224e-05, 1317000, 4.162998e-05, 1356000, 2.326474e-05, 1400000,
                             2.539037e-05, 1500000, 5.488273e-05, 1850000, 0.0001650281, 2354000, 0.0001817054, 2479000,
                             3.724223e-05, 3000000, 0.0001284582, 4304000, 0.0001884708, 4800000, 3.942526e-05, 6434000,
                             6.737429e-05, 8187300, 2.228427e-05, 1e+07, 6.200051e-06, 1.284e+07, 1.935622e-06,
                             1.384e+07, 1.369027e-07, 1.455e+07, 4.962589e-08, 1.5683e+07, 3.75955e-08, 1.7333e+07,
                             2.094257e-08, 2e+07, 8.083706e-09]

    for cov_energy_group in cov_groups:
        run_cover_chain(endf_file, mts, [300],
                        output_dir=output_dir, cov_energy_groups=cov_energy_group, iwt_fluxweight=6 ,
                        user_flux_weight_vals=user_flux_weight_vals, nu=nu, chi=chi, use_temp_folder=False)
        for mt in mts:
            process_cov_to_h5(output_dir, zaid, mt, boxer_matrix_name='covr_300.txt_{mt}_matrix.txt',
                              output_h5_format='u235_102_{0}g_cov.h5')

        if nu:
            process_cov_to_h5(output_dir, zaid, 452, boxer_matrix_name='covr_nu_300.txt_{mt}_matrix.txt',
                              output_h5_format='u235_102_{0}g_cov.h5')

        if chi:
            process_cov_to_h5(output_dir, zaid, 18, boxer_matrix_name='covr_chi_300.txt_{mt}_matrix.txt',
                              output_h5_format='u235_102_{0}g_cov.h5', cov_in_boxer=True)
