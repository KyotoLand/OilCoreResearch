import numpy as np
import openpyxl
import pywt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


def calculate_wavelet_coeffs(data, wavelet='mexh'):
    # Apply CWT to each row
    coeffs = []
    widths = np.arange(1, 30)
    for d in data.values():
        cwt_matr, _ = pywt.cwt(d, widths, wavelet)
        coeffs.append(cwt_matr.ravel())
    X = np.array(coeffs)
    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def get_capillary_curves(data):
    capillary_curves_data = data.iloc[3:, 12:]
    res = {}
    for i, row in capillary_curves_data.iterrows():
        res[i] = list(row)
    return res


def get_capillary_curves_with_log_permeability(data):
    capillary_curves_data = data.iloc[3:, 13:]
    perm = get_permeability(data)
    res = {}
    for i, row in capillary_curves_data.iterrows():

        res[i] = list(row)
        res[i].append(np.log10(perm[i]))
    return res


def get_capillary_curves_with_terr(data):
    column_index = 0
    start_row = 3
    value_map = {'терригенныф': 0, 'карбонатный': 1}
    collector_type = np.array(get_values_from_excel('data.xlsx', column_index, start_row, value_map))
    capillary_curves_data = data.iloc[3:, 12:]
    res = {}
    for i, row in capillary_curves_data.iterrows():
        res[i] = list(row)
    iteration = 0
    output = {}
    for key in res.keys():
        if collector_type[iteration] == 0:
            output[key] = res[key]
        iteration += 1
    print(output)
    return output


def get_porosity_and_permeability(data):
    capillary_curves_data = data.iloc[3:, 12:14]
    res = {}
    for i, row in capillary_curves_data.iterrows():
        res[i] = list(row)
    return res


def get_porosity(data):
    capillary_curves_data = data.iloc[3:, 13:14]
    res = {}
    for i, row in capillary_curves_data.iterrows():
        res[i] = list(row)
    return res


def get_permeability(data):
    capillary_curves_data = data.iloc[3:, 12:13]
    res = {}
    for i, row in capillary_curves_data.iterrows():
        res[i] = list(row)
    return res


def read_xlsx_file(file_path):
    # Load the workbook into a Pandas dataframe
    df = pd.read_excel(file_path, sheet_name=0)

    # Return the dataframe
    return df


def save_clusters_pp_to_xlsx(values, labels, filepath):
    # Create a DataFrame with the values and their corresponding labels
    # Create a DataFrame with the features and cluster labels
    df = pd.DataFrame(values, columns=['permeability', 'porosity'])
    df['cluster'] = labels

    # Export the cluster information to an Excel file
    df.to_excel(filepath, index=False)


def get_pp_by_fields_from_xlsx(file_path):
    wb = openpyxl.load_workbook(file_path)
    sheet = wb.active
    data_dict = {}

    for row in sheet.iter_rows(min_row=5, values_only=True):
        group_name = row[1]
        group_values = [row[12], row[13]]

        if group_name in data_dict:
            data_dict[group_name].append(group_values)
        else:
            data_dict[group_name] = [group_values]

    return data_dict


def get_values_from_excel(filename, column_index, start_row, value_map):
    """
    Reads an excel file and creates a list of 0s and 1s based on the values in a specific column.

    Args:
    - filename (str): The name of the excel file to read.
    - column_index (int): The index of the column that contains the values to extract.
    - start_row (int): The index of the first row to extract values from (0-based).
    - value_map (dict): A dictionary that maps values in the column to 0s and 1s.

    Returns:
    - result (list): A list of 0s and 1s based on the values in the column.
    """
    # Read the excel file
    df = pd.read_excel(filename)

    # Extract values from the column starting from the specified row
    values = df.iloc[start_row:, column_index].tolist()

    # Create a list of 0s and 1s based on the values in the column
    result = []
    for val in values:
        if val in value_map:
            result.append(value_map[val])
        else:
            # Handle other values if necessary
            pass

    return result


def plot_capillary_curve(capillary_curve, name):
    y = [0, 0.005, 0.010, 0.015, 0.025, 0.050, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2]
    x = capillary_curve[2:]
    plt.plot(x, y, marker='o')
    plt.xlabel("Насыщенность порового пространства, %")
    plt.ylabel("Капиллярное давление, МПа")
    plt.title("Капиллярная кривая для образца доломита")
    plt.legend(["Пористость - " + str(capillary_curve[1]) + " %\n" +
               "Проницаемость - " + str(capillary_curve[0]) + " мД"])
    plt.grid()
    plt.show()
