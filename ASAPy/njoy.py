"""Module for generating njoy files.

From openmc

Under MIT License:

Copyright (c) 2011-2018 Massachusetts Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

@veeshy added error module to generate covariances

"""

import argparse
from collections import namedtuple
from io import StringIO
import os
import shutil
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import sys
import tempfile

from ASAPy import endf


# For a given MAT number, give a name for the ACE table and a list of ZAID
# identifiers
ThermalTuple = namedtuple('ThermalTuple', ['name', 'zaids', 'nmix'])
_THERMAL_DATA = {
    1: ThermalTuple('hh2o', [1001], 1),
    2: ThermalTuple('parah', [1001], 1),
    3: ThermalTuple('orthoh', [1001], 1),
    5: ThermalTuple('hyh2', [1001], 1),
    7: ThermalTuple('hzrh', [1001], 1),
    8: ThermalTuple('hcah2', [1001], 1),
    10: ThermalTuple('hice', [1001], 1),
    11: ThermalTuple('dd2o', [1002], 1),
    12: ThermalTuple('parad', [1002], 1),
    13: ThermalTuple('orthod', [1002], 1),
    26: ThermalTuple('be', [4009], 1),
    27: ThermalTuple('bebeo', [4009], 1),
    31: ThermalTuple('graph', [6000, 6012, 6013], 1),
    33: ThermalTuple('lch4', [1001], 1),
    34: ThermalTuple('sch4', [1001], 1),
    37: ThermalTuple('hch2', [1001], 1),
    39: ThermalTuple('lucite', [1001], 1),
    40: ThermalTuple('benz', [1001, 6000, 6012], 2),
    41: ThermalTuple('od2o', [8016, 8017, 8018], 1),
    43: ThermalTuple('sisic', [14028, 14029, 14030], 1),
    44: ThermalTuple('csic', [6000, 6012, 6013], 1),
    46: ThermalTuple('obeo', [8016, 8017, 8018], 1),
    47: ThermalTuple('sio2-a', [8016, 8017, 8018, 14028, 14029, 14030], 3),
    48: ThermalTuple('uuo2', [92238], 1),
    49: ThermalTuple('sio2-b', [8016, 8017, 8018, 14028, 14029, 14030], 3),
    50: ThermalTuple('oice', [8016, 8017, 8018], 1),
    52: ThermalTuple('mg24', [12024], 1),
    53: ThermalTuple('al27', [13027], 1),
    55: ThermalTuple('yyh2', [39089], 1),
    56: ThermalTuple('fe56', [26056], 1),
    58: ThermalTuple('zrzrh', [40000, 40090, 40091, 40092, 40094, 40096], 1),
    59: ThermalTuple('cacah2', [20040, 20042, 20043, 20044, 20046, 20048], 1),
    75: ThermalTuple('ouo2', [8016, 8017, 8018], 1),
}

energy_groups = [2.0000E7, 1.7333E7, 1.5683E7, 1.4550E7, 1.3840E7, 1.2840E7, 1.0000E7, 8.1873E6, 6.4340E6, 4.8000E6, 4.3040E6, 3.0000E6, 2.4790E6, 2.3540E6, 1.8500E6, 1.5000E6, 1.4000E6, 1.3560E6, 1.3170E6, 1.2500E6, 1.2000E6, 1.1000E6, 1.0100E6, 9.2000E5, 9.0000E5, 8.7500E5, 8.6110E5, 8.2000E5, 7.5000E5, 6.7900E5, 6.7000E5, 6.0000E5, 5.7300E5, 5.5000E5, 4.9952E5, 4.7000E5, 4.4000E5, 4.2000E5, 4.0000E5, 3.3000E5, 2.7000E5, 2.0000E5, 1.5000E5, 1.2830E5, 1.0000E5, 8.5000E4, 8.2000E4, 7.5000E4, 7.3000E4, 6.0000E4, 5.2000E4, 5.0000E4, 4.5000E4, 3.0000E4, 2.5000E4, 1.7000E4, 1.3000E4, 9.5000E3, 8.0300E3, 6.0000E3, 3.9000E3, 3.7400E3, 3.0000E3, 2.5800E3, 2.2900E3, 2.2000E3, 1.8000E3, 1.5500E3, 1.5000E3, 1.1500E3, 9.5000E2, 6.8300E2, 6.7000E2, 5.5000E2, 3.0500E2, 2.8500E2, 2.4000E2, 2.1000E2, 2.0750E2, 1.9250E2, 1.8600E2, 1.2200E2, 1.1900E2, 1.1500E2, 1.0800E2, 1.0000E2, 9.0000E1, 8.2000E1, 8.0000E1, 7.6000E1, 7.2000E1, 6.7500E1, 6.5000E1, 6.1000E1, 5.9000E1, 5.3400E1, 5.2000E1, 5.0600E1, 4.9200E1, 4.8300E1, 4.7000E1, 4.5200E1, 4.4000E1, 4.2400E1, 4.1000E1, 3.9600E1, 3.9100E1, 3.8000E1, 3.7000E1, 3.5500E1, 3.4600E1, 3.3750E1, 3.3250E1, 3.1750E1, 3.1250E1, 3.0000E1, 2.7500E1, 2.5000E1, 2.2500E1, 2.1000E1, 2.0000E1, 1.9000E1, 1.8500E1, 1.7000E1, 1.6000E1, 1.5100E1, 1.4400E1, 1.3750E1, 1.2900E1, 1.1900E1, 1.1500E1, 1.0000E1, 9.1000E0, 8.1000E0, 7.1500E0, 7.0000E0, 6.7500E0, 6.5000E0, 6.2500E0, 6.0000E0, 5.4000E0, 5.0000E0, 4.7500E0, 4.0000E0, 3.7300E0, 3.5000E0, 3.1500E0, 3.0500E0, 3.0000E0, 2.9700E0, 2.8700E0, 2.7700E0, 2.6700E0, 2.5700E0, 2.4700E0, 2.3800E0, 2.3000E0, 2.2100E0, 2.1200E0, 2.0000E0, 1.9400E0, 1.8600E0, 1.7700E0, 1.6800E0, 1.5900E0, 1.5000E0, 1.4500E0, 1.4000E0, 1.3500E0, 1.3000E0, 1.2500E0, 1.2250E0, 1.2000E0, 1.1750E0, 1.1500E0, 1.1400E0, 1.1300E0, 1.1200E0, 1.1100E0, 1.1000E0, 1.0900E0, 1.0800E0, 1.0700E0, 1.0600E0, 1.0500E0, 1.0400E0, 1.0300E0, 1.0200E0, 1.0100E0, 1.0000E0, 9.7500E-1, 9.5000E-1, 9.2500E-1, 9.0000E-1, 8.5000E-1, 8.0000E-1, 7.5000E-1, 7.0000E-1, 6.5000E-1, 6.2500E-1, 6.0000E-1, 5.5000E-1, 5.0000E-1, 4.5000E-1, 4.0000E-1, 3.7500E-1, 3.5000E-1, 3.2500E-1, 3.0000E-1, 2.7500E-1, 2.5000E-1, 2.2500E-1, 2.0000E-1, 1.7500E-1, 1.5000E-1, 1.2500E-1, 1.0000E-1, 9.0000E-2, 8.0000E-2, 7.0000E-2, 6.0000E-2, 5.0000E-2, 4.0000E-2, 3.0000E-2, 2.5300E-2, 1.0000E-2, 7.5000E-3, 5.0000E-3, 4.0000E-3, 3.0000E-3, 2.5000E-3, 2.0000E-3, 1.5000E-3, 1.2000E-3, 1.0000E-3, 7.5000E-4, 5.0000E-4, 1.0000E-4, 1.0000E-5]
energy_groups = energy_groups[-1::-1]

_TEMPLATE_RECONR = """
reconr / %%%%%%%%%%%%%%%%%%% Reconstruct XS for neutrons %%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npendf}
'{library} PENDF for {zsymam}'/
{mat} 2/
{error}/ err
'{library}: {zsymam}'/
'Processed by NJOY'/
0/
"""

_TEMPLATE_BROADR = """
broadr / %%%%%%%%%%%%%%%%%%%%%%% Doppler broaden XS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npendf} {nbroadr}
{mat} {num_temp} 0 0 0. /
{error}/ errthn
{temps}
0/
"""

_TEMPLATE_HEATR = """
heatr / %%%%%%%%%%%%%%%%%%%%%%%%% Add heating kerma %%%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {nheatr_in} {nheatr} /
{mat} 3 /
302 318 402 /
"""

_TEMPLATE_PURR = """
purr / %%%%%%%%%%%%%%%%%%%%%%%% Add probability tables %%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npurr_in} {npurr} /
{mat} {num_temp} 1 20 64 /
{temps}
1.e10
0/
"""

_TEMPLATE_ACER = """
acer / %%%%%%%%%%%%%%%%%%%%%%%% Write out in ACE format %%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {nacer_in} 0 {nace} {ndir}
1 0 1 .{ext} /
'{library}: {zsymam} at {temperature}'/
{mat} {temperature}
1 1/
/
"""

_THERMAL_TEMPLATE_THERMR = """
thermr / %%%%%%%%%%%%%%%% Add thermal scattering data (free gas) %%%%%%%%%%%%%%%
0 {nthermr1_in} {nthermr1}
0 {mat} 12 {num_temp} 1 0 {iform} 1 221 1/
{temps}
{error} {energy_max}
thermr / %%%%%%%%%%%%%%%% Add thermal scattering data (bound) %%%%%%%%%%%%%%%%%%
{nthermal_endf} {nthermr2_in} {nthermr2}
{mat_thermal} {mat} 16 {num_temp} {inelastic} {elastic} {iform} {natom} 222 1/
{temps}
{error} {energy_max}
"""

"""
mfcov int
    the cov mf # to read, one of: 31, 33, 34, 35 or 40. 33 default for NJOY but must be provided here
"""
_TEMPLATE_ERRORR = """
errorr / %%%%%%%%%%%%%%%%%%%%%%%%% Calc COV %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {nerr_in} 0 {nerr}/
{mat} 1/
0 {temperature}/
0 {mfcov}/
{cov_ngroups}/
{cov_group_bounds}/
"""

_TEMPLATE_COVR_FOR_PLOT = """
covr / %%%%%%%%%%%%%%%%%%%%%%%%% Create Cov Info (PLOT) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
{nerr} 0 {covr_plot_out}/
1/
/
1 2/
{mat} 1 {mat} 1/
{mat} 102 {mat} 102/
"""

_TEMPLATE_COVR_FOR_TEXT = """
covr / %%%%%%%%%%%%%%%%%%%%%%%%% Create Cov Info (DATA) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
{nerr} {covr_out}/
1/
/
/
{mat}/
"""

_TEMPLATE_VIEWR = """
viewr / %%%%%%%%%%%%%%%%%%%%%%%%% Create Plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
{covr_plot_out} {viewr_plot_out}/
"""

_THERMAL_TEMPLATE_ACER = """
acer / %%%%%%%%%%%%%%%%%%%%%%%% Write out in ACE format %%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {nthermal_acer_in} 0 {nace} {ndir}
2 0 1 .{ext}/
'{library}: {zsymam_thermal} processed by NJOY'/
{mat} {temperature} '{data.name}' /
{zaids} /
222 64 {mt_elastic} {elastic_type} {data.nmix} {energy_max} 2/
"""


def run(commands, tapein, tapeout, input_filename=None, stdout=False,
        njoy_exec='/Users/veeshy/projects/NJOY2016/bin/njoy'):
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


def make_pendf(filename, pendf='pendf', error=0.001, stdout=False):
    """Generate ACE file from an ENDF file

    Parameters
    ----------
    filename : str
        Path to ENDF file
    pendf : str, optional
        Path of pointwise ENDF file to write
    error : float, optional
        Fractional error tolerance for NJOY processing
    stdout : bool
        Whether to display NJOY standard output

    Raises
    ------
    subprocess.CalledProcessError
        If the NJOY process returns with a non-zero status

    """

    make_ace(filename, pendf=pendf, error=error, broadr=False,
             heatr=False, purr=False, acer=False, stdout=stdout)


def make_ace(filename, temperatures=None, ace='ace', xsdir='xsdir', pendf=None,
             error=0.001, broadr=True, heatr=True, purr=True, acer=True, errorr=True,
             **kwargs):
    """Generate incident neutron ACE file from an ENDF file

    Parameters
    ----------
    filename : str
        Path to ENDF file
    temperatures : iterable of float, optional
        Temperatures in Kelvin to produce ACE files at. If omitted, data is
        produced at room temperature (293.6 K).
    ace : str, optional
        Path of ACE file to write
    xsdir : str, optional
        Path of xsdir file to write
    pendf : str, optional
        Path of pendf file to write. If omitted, the pendf file is not saved.
    error : float, optional
        Fractional error tolerance for NJOY processing
    broadr : bool, optional
        Indicating whether to Doppler broaden XS when running NJOY
    heatr : bool, optional
        Indicating whether to add heating kerma when running NJOY
    purr : bool, optional
        Indicating whether to add probability table when running NJOY
    acer : bool, optional
        Indicating whether to generate ACE file when running NJOY
    errorr : bool, optional
        Indicating whether to generate covariances when running NJOY
    **kwargs
        Keyword arguments passed to :func:`openmc.data.njoy.run`

    Raises
    ------
    subprocess.CalledProcessError
        If the NJOY process returns with a non-zero status

    """
    ev = endf.Evaluation(filename)
    mat = ev.material
    zsymam = ev.target['zsymam']

    # Determine name of library
    library = '{}-{}.{}'.format(*ev.info['library'])

    if temperatures is None:
        temperatures = [293.6]
    num_temp = len(temperatures)
    temps = ' '.join(str(i) for i in temperatures)

    # Create njoy commands by modules
    commands = ""

    nendf, npendf = 20, 21
    tapein = {nendf: filename}
    tapeout = {}
    if pendf is not None:
        tapeout[npendf] = pendf

    # reconr
    commands += _TEMPLATE_RECONR
    nlast = npendf

    # broadr
    if broadr:
        nbroadr = nlast + 1
        commands += _TEMPLATE_BROADR
        nlast = nbroadr
        
    if errorr:
        cov_ngroups = len(energy_groups) - 1
        for i, temperature in enumerate(temperatures):
            nerr_in = nbroadr  # PENDF tape that was broadened to right temp
            nerr = nlast + 1
            mfcov = 33
            cov_group_bounds = [str(e) for e in energy_groups]
            # inefficient way to ensure we don't go past 80 chars for fortran limitations
            cov_group_bounds = ' \n'.join(cov_group_bounds)
            commands += _TEMPLATE_ERRORR.format(**locals())
            nlast += 1

            covr_out = nlast + 1
            nlast += 1
            commands += _TEMPLATE_COVR_FOR_TEXT.format(**locals())

            covr_plot_out = nlast + 1   # needed as input to viewr
            nlast += 1

            viewr_plot_out = nlast + 1
            nlast += 1

            commands += _TEMPLATE_COVR_FOR_PLOT.format(**locals())
            commands += _TEMPLATE_VIEWR.format(**locals())

    # heatr
    if heatr:
        nheatr_in = nbroadr
        nheatr = nlast + 1
        commands += _TEMPLATE_HEATR
        nlast = nheatr

    # purr
    if purr:
        npurr_in = nlast
        npurr = npurr_in + 1
        commands += _TEMPLATE_PURR
        nlast = npurr

    commands = commands.format(**locals())

    # acer
    if acer:
        nacer_in = nlast
        fname = '{}_{:.1f}'
        for i, temperature in enumerate(temperatures):
            # Extend input with an ACER run for each temperature
            nace = nacer_in + 1 + 2*i
            ndir = nace + 1
            ext = '{:02}'.format(i + 1)
            commands += _TEMPLATE_ACER.format(**locals())

            # Indicate tapes to save for each ACER run
            tapeout[nace] = fname.format(ace, temperature)
            tapeout[ndir] = fname.format(xsdir, temperature)
    commands += 'stop\n'
    run(commands, tapein, tapeout, **kwargs)

    if acer:
        with open(ace, 'w') as ace_file, open(xsdir, 'w') as xsdir_file:
            for temperature in temperatures:
                # Get contents of ACE file
                text = open(fname.format(ace, temperature), 'r').read()

                # If the target is metastable, make sure that ZAID in the ACE file reflects
                # this by adding 400
                if ev.target['isomeric_state'] > 0:
                    mass_first_digit = int(text[3])
                    if mass_first_digit <= 2:
                        text = text[:3] + str(mass_first_digit + 4) + text[4:]

                # Concatenate into destination ACE file
                ace_file.write(text)

                # Concatenate into destination xsdir file
                text = open(fname.format(xsdir, temperature), 'r').read()
                xsdir_file.write(text)

        # Remove ACE/xsdir files for each temperature
        for temperature in temperatures:
            os.remove(fname.format(ace, temperature))
            os.remove(fname.format(xsdir, temperature))


def make_ace_thermal(filename, filename_thermal, temperatures=None,
                     ace='ace', xsdir='xsdir', error=0.001, **kwargs):
    """Generate thermal scattering ACE file from ENDF files

    Parameters
    ----------
    filename : str
        Path to ENDF neutron sublibrary file
    filename_thermal : str
        Path to ENDF thermal scattering sublibrary file
    temperatures : iterable of float, optional
        Temperatures in Kelvin to produce data at. If omitted, data is produced
        at all temperatures given in the ENDF thermal scattering sublibrary.
    ace : str, optional
        Path of ACE file to write
    xsdir : str, optional
        Path of xsdir file to write
    error : float, optional
        Fractional error tolerance for NJOY processing
    **kwargs
        Keyword arguments passed to :func:`openmc.data.njoy.run`

    Raises
    ------
    subprocess.CalledProcessError
        If the NJOY process returns with a non-zero status

    """
    ev = endf.Evaluation(filename)
    mat = ev.material
    zsymam = ev.target['zsymam']

    ev_thermal = endf.Evaluation(filename_thermal)
    mat_thermal = ev_thermal.material
    zsymam_thermal = ev_thermal.target['zsymam']

    data = _THERMAL_DATA[mat_thermal]
    zaids = ' '.join(str(zaid) for zaid in data.zaids[:3])

    # Determine name of library
    library = '{}-{}.{}'.format(*ev_thermal.info['library'])

    # Determine if thermal elastic is present
    if (7, 2) in ev_thermal.section:
        elastic = 1
        mt_elastic = 223

        # Determine whether elastic is incoherent (0) or coherent (1)
        file_obj = StringIO(ev_thermal.section[7, 2])
        elastic_type = endf.get_head_record(file_obj)[2] - 1
    else:
        elastic = 0
        mt_elastic = 0
        elastic_type = 0

    # Determine number of principal atoms
    file_obj = StringIO(ev_thermal.section[7, 4])
    items = endf.get_head_record(file_obj)
    items, values = endf.get_list_record(file_obj)
    energy_max = values[3]
    natom = int(values[5])

    # Note that the 'iform' parameter is omitted in NJOY 99. We assume that the
    # user is using NJOY 2012 or later.
    iform = 0
    inelastic = 2

    # Determine temperatures from MF=7, MT=4 if none were specified
    if temperatures is None:
        file_obj = StringIO(ev_thermal.section[7, 4])
        endf.get_head_record(file_obj)
        endf.get_list_record(file_obj)
        endf.get_tab2_record(file_obj)
        params = endf.get_tab1_record(file_obj)[0]
        temperatures = [params[0]]
        for i in range(params[2]):
            temperatures.append(endf.get_list_record(file_obj)[0][0])

    num_temp = len(temperatures)
    temps = ' '.join(str(i) for i in temperatures)

    # Create njoy commands by modules
    commands = ""

    nendf, nthermal_endf, npendf = 20, 21, 22
    # todo cp to a tape file here
    tapein = {nendf: filename, nthermal_endf:filename_thermal}
    tapeout = {}

    # reconr
    commands += _TEMPLATE_RECONR
    nlast = npendf

    # broadr
    nbroadr = nlast + 1
    commands += _TEMPLATE_BROADR
    nlast = nbroadr

    # thermr
    nthermr1_in = nlast
    nthermr1 = nthermr1_in + 1
    nthermr2_in = nthermr1
    nthermr2 = nthermr2_in + 1
    commands += _THERMAL_TEMPLATE_THERMR
    nlast = nthermr2

    commands = commands.format(**locals())

    # acer
    nthermal_acer_in = nlast
    fname = '{}_{:.1f}'
    for i, temperature in enumerate(temperatures):
        # Extend input with an ACER run for each temperature
        nace = nthermal_acer_in + 1 + 2*i
        ndir = nace + 1
        ext = '{:02}'.format(i + 1)
        commands += _THERMAL_TEMPLATE_ACER.format(**locals())

        # Indicate tapes to save for each ACER run
        tapeout[nace] = fname.format(ace, temperature)
        tapeout[ndir] = fname.format(xsdir, temperature)
    commands += 'stop\n'
    run(commands, tapein, tapeout, **kwargs)

    with open(ace, 'w') as ace_file, open(xsdir, 'w') as xsdir_file:
        # Concatenate ACE and xsdir files together
        for temperature in temperatures:
            text = open(fname.format(ace, temperature), 'r').read()
            ace_file.write(text)

            text = open(fname.format(xsdir, temperature), 'r').read()
            xsdir_file.write(text)

    # Remove ACE/xsdir files for each temperature
    for temperature in temperatures:
        os.remove(fname.format(ace, temperature))
        os.remove(fname.format(xsdir, temperature))
        
if __name__ == "__main__":
    make_ace('../data/e71/tape20', temperatures=[300], ace='ace', xsdir='xsdir', pendf=None,
                 error=0.001, broadr=True, heatr=False, purr=False, acer=False, errorr=True, **{'input_filename': '../data/cov_w184.i'})
    