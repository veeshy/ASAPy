import re
from pyne import ace
import numpy as np
import os

class AceEditor:
    """
    Loads a table from an ace file. If there is only one table, loads that table
    """

    def __init__(self, ace_path, specific_table=None):
        ace_path = os.path.expanduser(ace_path)
        libFile = ace.Library(ace_path)
        libFile.read()

        if specific_table:
            table = libFile.find_table(specific_table)
        else:
            tables = libFile.tables
            if len(tables) > 1:
                raise Exception('Multiple libraries found, please specify specific_table')

            table = libFile.find_table(list(tables.keys())[0])

        self.ace_path = ace_path
        self.table = table
        self.energy = table.energy
        self.all_mts = self.table.reactions.keys()
        # store the original sigma so it will be easy to retrieve it later
        self.original_sigma = {}
        self.adjusted_mts = set()

    def get_sigma(self, mt):
        """
        Grabs sigma from the table
        Parameters
        ----------
        mt : int

        Returns
        -------
        pyne.ace.NeutronTable.sigma
            Mutable nd.array

        """
        rx = self.table.find_reaction(mt)

        return rx.sigma

    def set_sigma(self, mt, sigma):
        """
        Sets the mt sigma and adds the mt to the changed sigma set
        Parameters
        ----------
        mt : int
        sigma : nd.array
        """

        if len(sigma) != len(self.energy):
            raise Exception('Length of sigma provided does not match energy bins got: {0}, needed: {1}'.format(len(sigma), len(self.energy)))

        current_sigma = self.get_sigma(mt)
        if mt not in self.original_sigma.keys():
            self.original_sigma[mt] = current_sigma.copy()

        self.table.find_reaction(mt).sigma = sigma

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
        # potential for round-off error but each mt is summed indvidually so order does not matter.
        # if a sum mt does not exist, nothing happens

        # considered redundant, may not need to adjust this if your program does not use it
        # 4, 27, 101, 3, 4, 1

        # mt_4 particularly does not define xsec on the union energy grid so summing it is non-trivial
        mt_4 = list(range(50, 92))
        mt_16 = list(range(875, 892))
        mt_18 = [19, 20, 21, 38]

        mt_103 = list(range(600, 650))
        mt_104 = list(range(650, 700))
        mt_105 = list(range(700, 750))
        mt_106 = list(range(750, 800))
        mt_107 = list(range(800, 850))
        mt_101 = [102, *mt_103, *mt_104, *mt_105, *mt_106, *mt_107, *list(range(108, 118)), 155, 182, 191, 192, 193,
                  197]  # contains 103, 104, 105, 106, 107
        mt_27 = [*mt_18, *mt_101]  # has 18 and 101 which are sums themselves

        mt_3 = [*mt_4, 5, 11, *mt_16, 17, *mt_18, *list(range(22, 27)), *list(range(28, 38)), 41, 42, 44, 45, *mt_101,
                152, 153, 154, *list(range(156, 182)),
                *list(range(183, 191)), 194, 195, 196, 198, 199, 200]  # contains 4 which is a sum

        mt_1 = [2, *mt_3]  # contains 3 which is a sum

        sum_mts_list = [mt_4, mt_16, mt_18, mt_103, mt_104, mt_105, mt_106, mt_107, mt_101, mt_27, mt_3, mt_1]
        sum_mts = [4, 16, 18, 103, 104, 105, 106, 107, 101, 27, 3, 1]

        for sum_mt, mts_in_sum in zip(sum_mts, sum_mts_list):
            # ensure the sum'd mt is present before trying to set it
            if sum_mt in self.all_mts:
                sum_mts_present = self._check_if_mts_present(mts_in_sum)
                if sum_mts_present:
                        # check if MT was adjusted before re-summing
                        mt_adjusted_check = self._check_if_mts_present(self.adjusted_mts, compare_to=sum_mts_present)

                        if mt_adjusted_check:
                            # re-write this mt with the constituent mts summed
                            sigmas = np.array([self.get_sigma(mt) for mt in sum_mts_present])
                            # sum all rows together
                            try:
                                new_sum = sigmas.sum(axis=0)
                            except ValueError:
                                raise ValueError("Could not sum the xsec's in mt={0}. Note: MTs 1, 3, 4, 27, 101 are "
                                                 "considered redundant, perhaps you don't need to apply the sum rule.\n\n"
                                                 "MTs in this sum that are in this ACE file:\n{1}".format(sum_mt, sum_mts_present))
                            self.set_sigma(sum_mt, new_sum)
                            print("Adjusting {0} due to ENDF sum rules".format(sum_mt))


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

    def replace_array(self, array, replace_with):
        """
        Replaces array with replace_with in the currently read ace file

        Parameters
        ----------
        array : list-like
        replace_with : list-like
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
        self.single_line = self.single_line.replace(self.single_line[first_idx_of_data: last_idx_of_data], replace_with_line)

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

    # read then write the ace file for testing (should be the same)
    #wa = WriteAce('/Users/veeshy/MCNP6/MCNP_DATA/xdata/endf71x/W/74184.710nc')
    #wa.write_ace('/Users/veeshy/MCNP6/MCNP_DATA/xdata/endf71x/W/test_74184.710nc')


    wa = AceEditor('~/MCNP6/MCNP_DATA/xdata/endf71x/U/92235.710nc')
    wa.adjusted_mts.add(51)
    wa.apply_sum_rules()