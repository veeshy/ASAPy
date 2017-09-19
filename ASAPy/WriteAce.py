import re

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
        original_str_formatted = [format(s, formatting) for s in st]
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