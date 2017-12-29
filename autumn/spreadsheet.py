

from xlrd import open_workbook
from numpy import nan
import numpy
import os
import tool_kit


''' static function '''


def parse_year_data(year_data, blank, end_column):
    """
    Code to parse rows of data that are years.

    Args:
        year_data: The row to parse
        blank: A value for blanks to be ignored
        end_column: Column to end at
    """

    year_data = tool_kit.replace_specified_value(year_data, nan, blank)
    assumption_val, year_vals = year_data[-1], year_data[:end_column]
    if tool_kit.is_all_same_value(year_vals, nan):
        return [assumption_val]
    else:
        return year_vals


''' spreadsheet reader object '''


class SpreadsheetReader:
    """
    The master spreadsheet reader that now subsumes all of the previous readers (which had been structured as one
    sheet reader per spreadsheet to be read. Now the consistently required data is indexed from dictionaries in
    instantiation before the row or column reading methods are called as required with if/elif statements to use the
    appropriate reading approach.
    """

    def __init__(self, country_to_read, purpose):
        """
        Use the "purpose" input to index static dictionaries to set basic reader characteristics.

        Args:
            country_to_read: The adapted country name
            purpose: String that defines the spreadsheet type to be read
        """

        # file and sheet names
        specific_sheet_names \
            = {'default_constants': 'xls/data_default.xlsx',
               'country_constants': 'xls/data_' + country_to_read.lower() + '.xlsx',
               'default_programs': 'xls/data_default.xlsx',
               'country_programs': 'xls/data_' + country_to_read.lower() + '.xlsx'}
        specific_tab_names \
            = {'default_constants': 'constants',
               'country_constants': 'constants',
               'default_programs': 'time_variants',
               'country_programs': 'time_variants'}

        # starting positions
        sheets_starting_row_one \
            = ['gtb_2015', 'gtb_2016', 'default_constants', 'country_constants', 'notifications_2015', 'outcomes_2013',
               'laboratories_2014', 'laboratories_2015', 'laboratories_2016']
        start_rows \
            = {'life_expectancy_2014': 3,
               'life_expectancy_2015': 3,
               'diabetes': 2}
        start_cols \
            = {'bcg_2014': 4,
               'bcg_2015': 4,
               'bcg_2016': 4,
               'default_programs': 1,
               'country_programs': 1}
        columns_for_keys \
            = {'bcg_2014': 2,
               'bcg_2015': 2,
               'bcg_2016': 2,
               'rate_birth': 2}
        first_cells \
            = {'bcg_2014': 'WHO_REGION',
               'bcg_2015': 'WHO_REGION',
               'bcg_2016': 'WHO_REGION',
               'rate_birth': 'Series Name',
               'life_expectancy_2014': 'Country Name',
               'life_expectancy_2015': 'Country Name',
               'gtb_2015': 'country',
               'gtb_2016': 'country',
               'default_constants': 'program',
               'country_constants': 'program',
               'default_programs': 'program',
               'country_programs': 'program',
               'diabetes': u'Country/territory'}

        # spreadsheets to be read vertically
        vertical_sheets \
            = ['gtb_2015', 'gtb_2016', 'notifications_2014', 'notifications_2015', 'notifications_2016',
               'outcomes_2013', 'outcomes_2015', 'laboratories_2014', 'laboratories_2015', 'laboratories_2016']

        # adaptation of the country name to the sheet's approach to naming
        tb_adjustment_countries \
            = ['gtb_2015', 'gtb_2016', 'notifications_2014', 'notifications_2015', 'notifications_2016',
               'outcomes_2013', 'outcomes_2015']
        country_adjustment_types \
            = {'rate_birth': 'demographic',
               'life_expectancy_2014': 'demographic',
               'life_expectancy_2015': 'demographic'}

        self.purpose = purpose
        if purpose in country_adjustment_types:
            country_adjustment = country_adjustment_types[purpose]
        elif purpose in tb_adjustment_countries:
            country_adjustment = 'tb'
        else:
            country_adjustment = ''
        self.country_to_read = tool_kit.adjust_country_name(country_to_read, country_adjustment)
        self.filename = specific_sheet_names[purpose] if purpose in specific_sheet_names else 'xls/' + purpose + '.xlsx'
        self.tab_name = specific_tab_names[purpose] if purpose in specific_tab_names else purpose
        if purpose in sheets_starting_row_one:
            self.start_row = 1
        elif purpose in start_rows:
            self.start_row = start_rows[purpose]
        else:
            self.start_row = 0
        self.start_col = start_cols[purpose] if purpose in start_cols else 0
        self.column_for_keys = columns_for_keys[purpose] if purpose in columns_for_keys else 0
        self.first_cell = first_cells[purpose] if purpose in first_cells else 'country'
        self.horizontal = False if self.purpose in vertical_sheets else True
        self.indices, self.parlist, self.dictionary_keys, self.data, self.year_indices = [], [], [], {}, {}

    def read_data_list(self, workbook):
        """
        Short function to determine whether to read horizontally or vertically.

        Arg:
            workbook: The Excel workbook for interrogation
        """

        sheet = workbook.sheet_by_name(self.tab_name)
        if self.horizontal:
            for i_row in range(self.start_row, sheet.nrows): self.parse_row(sheet.row_values(i_row))
        else:
            for i_col in range(self.start_col, sheet.ncols): self.parse_col(sheet.col_values(i_col))
        return self.data

    def parse_row(self, row):
        """
        Method to read rows of the spreadsheets for sheets that read horizontally. Several different spreadsheet readers
        use this method, so there is a boolean loop to determine which approach to use according to the sheet name.

        Args:
            row: The row of data
        """

        # vaccination sheet
        if self.purpose == 'bcg_2016':
            if row[0] == self.first_cell:
                self.parlist = parse_year_data(row, '', len(row))
                for i in range(len(self.parlist)): self.parlist[i] = str(self.parlist[i])
            elif row[self.column_for_keys] == self.country_to_read:
                for i in range(self.start_col, len(row)):
                    if type(row[i]) == float: self.data[int(self.parlist[i])] = row[i]

        # demographics
        elif self.purpose in ['rate_birth', 'life_expectancy_2014', 'life_expectancy_2015']:
            if row[0] == self.first_cell:
                for i in range(len(row)): self.parlist.append(row[i][:4])
            elif row[self.column_for_keys] == self.country_to_read:
                for i in range(4, len(row)):
                    if type(row[i]) == float: self.data[int(self.parlist[i])] = row[i]

        # constants
        elif self.purpose in ['default_constants', 'country_constants']:

            # age breakpoints
            if row[0] == 'age_breakpoints':
                self.data[str(row[0])] = []
                for i in range(1, len(row)):
                    if row[i]: self.data[str(row[0])].append(int(row[i]))

            # empty entries
            elif row[1] == '':
                pass

            # country or integration method
            elif row[0] in ['country', 'integration']:
                self.data[str(row[0])] = str(row[1])

            # model strata or fitting method
            elif row[0][:2] == 'n_' or 'fitting' in row[0]:
                self.data[str(row[0])] = int(row[1])

            # for the calendar year times
            elif 'time' in row[0] or 'smoothness' in row[0]:
                self.data[str(row[0])] = float(row[1])

            # all instructions around outputs, plotting and spreadsheet/document writing
            elif 'output_' in row[0] or row[0][:3] == 'is_' or row[0][:12] == 'comorbidity_':
                self.data[str(row[0])] = bool(row[1])

            # parameter values
            else:
                self.data[str(row[0])] = row[1]

            # uncertainty parameters
            # not sure why this if statement needs to be split and written like this, but huge bugs occur if it isn't
            if len(row) >= 4:
                # if an entry present in the second column, then it is a parameter that can be modified in uncertainty
                if row[2] != '':
                    self.data[str(row[0]) + '_uncertainty'] = {'point': row[1], 'lower': row[2], 'upper': row[3]}

        # time-variant programs and interventions
        elif self.purpose in ['default_programs', 'country_programs']:
            if row[0] == self.first_cell:
                self.parlist = parse_year_data(row, '', len(row))
            else:
                self.data[str(row[0])] = {}
                for i in range(self.start_col, len(row)):
                    parlist_item_string = str(self.parlist[i])
                    if ('19' in parlist_item_string or '20' in parlist_item_string) and row[i] != '':
                        self.data[row[0]][int(self.parlist[i])] = row[i]
                    elif row[i] != '':
                        self.data[str(row[0])][str(self.parlist[i])] = row[i]

        # diabetes
        elif self.purpose == 'diabetes':
            if row[0] == self.first_cell:
                self.dictionary_keys.append(row)
            elif row[0] == self.country_to_read:
                for i in range(len(self.dictionary_keys)):
                    if self.dictionary_keys[i][:28] == u'Diabetes national prevalence':
                        self.data['comorb_prop_diabetes'] = float(row[i][:row[i].find('\n')]) / 1e2

        # other sheets, such as strategy and mdr
        else:

            # create the list to turn in to dictionary keys later
            if row[0] == self.first_cell:
                self.dictionary_keys = row

            # populate when country to read is encountered
            elif row[0] == self.country_to_read:
                for i in range(len(self.dictionary_keys)): self.data[self.dictionary_keys[i]] = row[i]

    def parse_col(self, col):
        """
        Read columns of data. Note that all the vertically oriented spreadsheets currently run off the same reading
        method, so there is not need for an outer if/elif statement to determine which approach to use - this is likely
        to change.

        Args:
            col: The column of data
        """

        col = tool_kit.replace_specified_value(col, nan, '')

        # country column (the first one)
        if col[0] == self.first_cell:

            # find the indices for the country
            for i in range(len(col)):
                if col[i] == self.country_to_read: self.indices.append(i)

        # skip some irrelevant columns
        elif 'iso' in col[0] or 'g_who' in col[0] or 'source' in col[0]:
            pass

        # year column
        elif col[0] == 'year':
            self.year_indices = {int(col[i]): i for i in self.indices}

        # all other columns
        else:
            self.data[str(col[0])] = {}
            for year in self.year_indices:
                if not numpy.isnan(col[self.year_indices[year]]): self.data[col[0]][year] = col[self.year_indices[year]]


''' master function to call readers '''


def read_input_data_xls(from_test, sheets_to_read, country):
    """
    Compile sheet readers into a list according to which ones have been selected.
    Note that most readers now take the country in question as an input, while only the fixed parameters sheet reader
    does not.

    Args:
        from_test: Whether being called from the directory above
        sheets_to_read: A list containing the strings that are also the 'keys' attribute of the reader
        country: Country being read
    Returns:
        A single data structure containing all the data to be read (by calling the read_xls_with_sheet_readers method)
    """

    # add sheet readers as required
    sheet_readers, data_read_from_sheets = [], {}
    available_sheets \
        = ['default_constants', 'bcg_2014', 'bcg_2015', 'bcg_2016', 'rate_birth', 'life_expectancy_2014',
           'life_expectancy_2015', 'country_constants', 'default_programs', 'country_programs', 'notifications_2014',
           'notifications_2015', 'notifications_2016', 'outcomes_2013', 'outcomes_2015', 'mdr_2014', 'mdr_2015',
           'mdr_2016', 'laboratories_2014', 'laboratories_2015', 'laboratories_2016', 'strategy_2014', 'strategy_2015',
           'strategy_2016', 'diabetes', 'gtb_2015', 'gtb_2016', 'latent_2016', 'tb_hiv_2016']
    for sheet_name in available_sheets:
        if sheet_name in sheets_to_read: sheet_readers.append(SpreadsheetReader(country, sheet_name))

    for reader in sheet_readers:

        # if being run from the directory above
        if from_test: reader.filename = os.path.join('autumn/', reader.filename)

        # check that the spreadsheet to be read exists
        try:
            print('Reading file ' + reader.filename)
            workbook = open_workbook(reader.filename)

        # if sheet unavailable, warn of issue
        except:
            print('Unable to open spreadsheet ' + reader.filename)

        # read the sheet according to reading orientation
        else:
            data_read_from_sheets[reader.purpose] = reader.read_data_list(workbook)
    return data_read_from_sheets

