import re
from .data import neutron
import numpy as np
import os
import copy

class AceEditor:
    """
    Loads a table from an ace file. If there is only one table, loads that table.

    Uses openmc data readers as backend, the openmc reader is much more powerful but we only need a subset of the reader so it lives in this class

    Parameters
    ----------
    ace_path : str
    temperature : str or float
        The temperature to get from the ace file, str ends with K or float will get K added to it, must be exactly what is on the ace file

    """

    def __init__(self, ace_path, temperature=None):

        ace_path = os.path.expanduser(ace_path)

        self.table = neutron.IncidentNeutron.from_ace(ace_path)


        temperatures = self.table.temperatures
        if not temperature:
            if len(temperatures) > 1:
                raise Exception('Multiple temperature found, please specify temperature')
            else:
                temperature = temperatures[0]
        else:
            temperature = str(temperature)
            if not temperature.endswith('K'):
                # try to see if the user left off "K"
                temperature += 'K'

            if temperature not in list(temperature):
                raise Exception("Could not find temperature {0} in ace file".format(temperature))

        self.temperature = temperature

        self.ace_path = ace_path
        self.all_mts = list(self.table.reactions.keys())
        # store the original sigma so it will be easy to retrieve it later
        self.original_sigma = {}
        self.adjusted_mts = set()

    def get_energy(self, mt):
        """
        Gets evaluated energies for mt. Mostly lets us handle non reaction type energies like nu-bar.

        Parameters
        ----------
        mt

        Returns
        -------
        list
            Energies where mt is defined (eV)

        """

        if mt in [452]:
            energy, _ = self.get_nu_distro()
        else:
            energy = self.table.energy[self.temperature]

        return energy

    def get_chi_distro(self, mt=18):
        """
        Gets the energy distribution from mt.

        Gets the prompt chi

        Returns
        -------
        energy_in : ndarray
            The incident neutron energy
        energy_out : list of ndarray
            The out neutron energy
        pdf : list of ndarray
            The PDF for the probability distribution of output energies
        cdf : list of ndarray
            The CDF for the probability distribution of output energies
        """
        fission_present = self._check_if_mts_present([mt])

        if fission_present:
            fission_chi_prompt_product = self.table.reactions[mt].products[0]
            fission_chi_prompt = fission_chi_prompt_product.distribution[0].energy

            if 'prompt' not in fission_chi_prompt_product.__repr__():
                raise Exception("Prompt fission chi not found on mt {0} of {1}".format(mt, self.ace_path))

            fission_chi_prompt_energy = fission_chi_prompt.energy
            fission_chi_prompt_energy_out = []
            fission_chi_prompt_energy_p = []
            fission_chi_prompt_energy_c = []
            for distro in fission_chi_prompt.energy_out:
                fission_chi_prompt_energy_out.append(distro.x)
                fission_chi_prompt_energy_p.append(distro.p)
                fission_chi_prompt_energy_c.append(distro.c)


            # only allow LAW=4, not sure how to check this with openmc..

        else:
            raise IndexError("No fission MT={mt} present on ace file {ace}".format(mt=mt, ace=self.ace_path))

        return fission_chi_prompt_energy, fission_chi_prompt_energy_out, fission_chi_prompt_energy_p, fission_chi_prompt_energy_c

    def get_nu_distro(self):
        """
        Finds \chi_T table.
        Raises
        ------
        AttributeError
            If the ACE file does ot contain nu_t_type
        TypeError
            If the table is not stored as tabular (rather than polynomial)
        Returns
        -------
        np.array
            Incident neutron energy causing fission
        np.array
            Total number of neutrons emitted per fission
        """

        try:
            fiss = self.table.reactions[18]
        except:
            raise KeyError("Fission MT=18 not present in this ACE " + self.ace_path)

        prod = fiss.derived_products[0]

        label = prod.__repr__()
        if 'tabulated' not in label:
            raise TypeError(
                "Fission nu distribition is not stored in the tabular format (ASAPy does not support poly) " + self.ace_path)

        nu_t_e = prod.yield_.x
        nu_t_value = prod.yield_.y

        return nu_t_e, nu_t_value



    def get_sigma(self, mt, at_energies=None):
        """
        Grabs sigma from the table
        Parameters
        ----------
        mt : int

        Returns
        -------
        np.array
            xsec values

        """

        # handle nu differently than other xsec
        if mt in [452]:
            _, sigma = self.get_nu_distro()
        else:
            try:
                rx = self.table.reactions[mt].xs[self.temperature]
            except KeyError:
                raise KeyError("Could not find mt={0} in ace file".format(mt))

            if at_energies is None:
                sigma = copy.deepcopy(rx.y)
            else:
                sigma = copy.deepcopy(rx(at_energies))

        return sigma

    def set_sigma(self, mt, sigma):
        """
        Sets the mt sigma and adds the mt to the changed sigma set
        Parameters
        ----------
        mt : int
        sigma : nd.array
        """

        energy = self.get_energy(mt)
        if len(sigma) != len(energy):
            raise IndexError('Length of sigma provided does not match energy bins got: {0}, needed: {1}'.format(len(sigma), len(energy)))

        current_sigma = self.get_sigma(mt)
        if mt not in self.original_sigma.keys():
            self.original_sigma[mt] = current_sigma.copy()

        if mt in [452]:
            self.table.reactions[18].derived_products[0].yield_.y = sigma
        else:
            self.table.reactions[mt].xs[self.temperature].y = sigma

        self.adjusted_mts.add(mt)


    def apply_sum_rules(self):
        """
        Applies ENDF sum rules to all MTs. MTs adjusted in place

        Assumes neutron

        References
        ----------
        @trkov2012endf : ENDF6 format manual

        Questions
        ---------
        if a summed MT is not present, but some of it's constiuents are, and that summed mt that is not present
        is present in another MT sum, should the not present MTs constituents be included in the sum?
        --I think so: if we don't have MT18 but have 19, it should still be in the total xsec
        """

        # the sum rules
        # potential for round-off error but each mt is summed individually so order does not matter.
        # if a sum mt does not exist, nothing happens

        mt_103 = list(range(600, 650))
        mt_104 = list(range(650, 700))
        mt_105 = list(range(700, 750))
        mt_106 = list(range(750, 800))
        mt_107 = list(range(800, 850))
        mt_101 = [102, *mt_103, *mt_104, *mt_105, *mt_106, *mt_107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 155,
                  182, 191, 192, 193, 197]
        mt_18 = [19, 20, 21, 38]
        mt_27 = [*mt_18, *mt_101]  # has 18 and 101 which are sums themselves

        mt_16 = list(range(875, 892))
        mt_4 = list(range(50, 92))
        # mt_3 = [*mt_4, 5, 11, *mt_16, 17, *list(range(22, 26)), *mt_27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 41, 42, 44,
        #         45, 152, 153, 154, *list(range(156, 182)), *list(range(183, 191)), 194, 195, 196, 198, 199, 200]

        # according to https://t2.lanl.gov/nis/endf/mts.html
        mt_3 = [4, 5, 11, 16, 17, 18, *list(range(22, 27)), *list(range(28, 38)), 41, 42, *list(range(44, 46)),
                *list(range(102, 118))]

        mt_1 = [2, *mt_3]  # contains 3 which is a sum

        # go from highest to lowest since lower ENDF #'s are not in higher #'s, except 18 which is in 27 so do 18 first
        sum_mts_list = [mt_107, mt_106, mt_105, mt_104, mt_103, mt_101, mt_18, mt_27, mt_16, mt_4, mt_3, mt_1]
        sum_mts = [107, 106, 105, 104, 103, 101, 18, 27, 16, 4, 3, 1]

        for sum_mt, mts_in_sum in zip(sum_mts, sum_mts_list):
            # ensure the sum'd mt is present before trying to set it
            if sum_mt in self.all_mts:
                sum_mts_present = self._check_if_mts_present(mts_in_sum)
                if sum_mts_present:
                        # check if MT was adjusted before re-summing
                        mt_adjusted_check = self._check_if_mts_present(self.adjusted_mts, compare_to=sum_mts_present)

                        if mt_adjusted_check:
                            # re-write this mt with the constituent mts summed
                            energies = self.get_energy(1)
                            sigmas = np.array([self.get_sigma(mt, at_energies=energies) for mt in sum_mts_present])
                            # sum all rows together
                            try:
                                new_sum = sigmas.sum(axis=0)
                            except ValueError:
                                raise ValueError("Could not sum the xsec's in mt={0}. Note: MTs 1, 3, 4, 27, 101 are "
                                                 "considered redundant, perhaps you don't need to apply the sum rule.\n\n"
                                                 "MTs in this sum that are in this ACE file:\n{1}".format(sum_mt, sum_mts_present))
                            self.set_sigma(sum_mt, new_sum)
                            self.adjusted_mts.add(sum_mt)


    def _check_if_mts_present(self, mt_list, compare_to=None):
        """
        Find what elements of mt_list are in self.all_mts
        Parameters
        ----------
        mt_list
            List of mt's to check for

        Returns
        -------
        list
            List of mt's present
        list
            List to check if mt's present in

        """
        if compare_to is None:
            compare_to = self.all_mts

        search_set = set(mt_list)
        search_in_set = set(compare_to)

        return list(search_in_set.intersection(search_set))


class WriteAce:
    """
    A simplistic way of modifying an existing ace file. Essentially a find and replace.
    """

    def __init__(self, ace_path):

        self.single_line, self.header_lines = self._read_ace_to_a_line(ace_path)

    def _read_ace_to_a_line(self, ace_path):
        """
        Read the ace file to adjust to a single line with no spaces
        """
        ace_path = os.path.expanduser(ace_path)
        with open(ace_path, 'r') as f:
            lines = f.readlines()

        # new type header: 2.0.0. (decimal in 2nd place of first word)
        number_of_header_lines = [15 if lines[0].split()[0][1] == '.' else 12][0]

        header_lines = lines[0:number_of_header_lines]

        # convert all lines to a single string for searching.
        # replace newlines with spaces, then remove all more than single spaces

        single_line = ''
        for l in lines[number_of_header_lines:]:
            single_line += l.replace('\n', ' ')

        single_line = re.sub(' +', ' ', single_line)

        return single_line, header_lines

    def format_mcnp(self, array, formatting='.11E'):
        """
        Format values in array based on MCNP formatting

        Parameters
        ----------
        array : list-like
        Returns
        -------
        list
        """
        original_str_formatted = [format(s, formatting) for s in array]
        original_str_line = ' '.join(original_str_formatted)

        return original_str_line

    def replace_array(self, array, replace_with, max_replaces=1):
        """
        Replaces array with replace_with in the currently read ace file

        Parameters
        ----------
        array : list-like
        replace_with : list-like
        max_replaces : int
            Max number of times to replace array found
        """

        if len(array) != len(replace_with):
            raise Exception("Arrays must be equal length for replacement.")

        original_str_line = self.format_mcnp(array)

        try:
            first_idx_of_data = self.single_line.index(original_str_line)
        except ValueError:
            raise ValueError("Inputted string not found in the ace file.")

        last_idx_of_data = first_idx_of_data + len(original_str_line)

        replace_with_line = self.format_mcnp(replace_with)
        self.single_line = self.single_line.replace(self.single_line[first_idx_of_data: last_idx_of_data],
                                                    replace_with_line, max_replaces)

    def write_ace(self, ace_path_to_write):

        split_all_data_str = self.single_line.split()
        ace_path_to_write = os.path.expanduser(ace_path_to_write)

        with open(ace_path_to_write, 'w') as f:
            # write the header back to the file
            [f.write(line) for line in self.header_lines]

            values_printed_coutner = 0
            for value in split_all_data_str:
                f.write('{0:>20}'.format(value))
                values_printed_coutner += 1
                if values_printed_coutner == 4:
                    f.write('\n')
                    values_printed_coutner = 0

            # ensure a new line at EOF
            if values_printed_coutner < 4:
                f.write('\n')

if __name__ == "__main__":
    pass
