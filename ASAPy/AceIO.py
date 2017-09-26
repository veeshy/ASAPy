import re
from pyne import ace
import numpy as np

class AceEditor:
    """
    Loads a table from an ace file. If there is only one table, loads that table
    """

    def __init__(self, ace_path, specific_table=None):
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
        self.changed_mts = set()

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
        current_sigma = sigma

        self.changed_mts.add(mt)

    def apply_sum_rules(self):
        """
        Applies ENDF sum rules to all MTs. MTs adjusted in place

        References
        ----------
        @trkov2012endf : ENDF6 format manual
        """

        # the sum rules
        # perform the sums in the exact order listed below to account for when a sum may appear in another sum

        mt_4 = list(range(50, 92))
        mt_3 = [4, 5, 11, 16, 17, *list(range(22, 38)), 41, 42, 44, 45, 152, 153, 154, *list(range(156, 182)),
                *list(range(183, 191)), 194, 195, 196, 198, 199, 200]  # contains 4 which is a sum

        mt_16 = list(range(875, 892))

        mt_18 = [19, 20, 21, 38]

        mt_103 = list(range(600, 650))
        mt_104 = list(range(650, 700))
        mt_105 = list(range(700, 750))
        mt_106 = list(range(750, 800))
        mt_107 = list(range(800, 850))
        mt_101 = [*list(range(102, 118)), 155, 182, 191, 192, 193, 197]  # contains 103, 104, 105, 106, 107
        mt_27 = [18, 101]  # has 18 and 101 which are sums themselves

        mt_1 = [2, 3]  # contains 3 which is a sum

        sum_mts_list = [mt_4, mt_3, mt_16, mt_18, mt_103, mt_104, mt_105, mt_106, mt_107, mt_101, mt_27, mt_1]
        sum_mts = [4, 3, 16, 18, 103, 104, 105, 106, 107, 101, 27, 1]

        for sum_mt, mts_in_sum in zip(sum_mts, sum_mts_list):

            # ensure the sum'd mt is present before trying to set it
            if sum_mt in self.all_mts:
                sum_mts_present = self._check_if_mts_present(mts_in_sum)
                if sum_mts_present:
                        # re-write this mt with the constituent mts summed
                        sigmas = np.array([self.get_sigma(mt) for mt in sum_mts_present])
                        # sum all rows together
                        new_sum = sigmas.sum(axis=0)
                        self.set_sigma(sum_mt, new_sum)
                        print("Adjusting {0} due to ENDF sum rules".format(sum_mt))


    def _check_if_mts_present(self, mt_list):
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

        """

        search_set = set(mt_list)
        search_in_set = set(self.all_mts)

        return list(search_in_set.intersection(search_set))




class WriteAce:
    """
    A simplistic way of modifying an existing ace file. Essentially a find and replace.
    """

    def __init__(self, ace_path):

        self.lines, self.header_lines = self._read_ace_to_a_line(ace_path)

    def _read_ace_to_a_line(self, ace_path):
        """
        Read the ace file to adjust to a single line with no spaces
        """

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
            raise Exception("Arrays must be equal lenght for replacement.")

        original_str_line = self.format_mcnp(array)

        try:
            first_idx_of_data = self.lines.index(original_str_line)
        except ValueError:
            raise ValueError("Inputted string not found in the ace file.")

        last_idx_of_data = first_idx_of_data + len(original_str_line)

        replace_with_line = self.format_mcnp(replace_with)
        self.single_line = self.single_line.replace(self.single_line[first_idx_of_data: last_idx_of_data], replace_with_line)

    def write_ace(self, ace_path_to_write):

        split_all_data_str = self.lines.split()

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
    wa = WriteAce('/Users/veeshy/MCNP6/MCNP_DATA/xdata/endf71x/W/74184.710nc')
    wa.write_ace('/Users/veeshy/MCNP6/MCNP_DATA/xdata/endf71x/W/test_74184.710nc')