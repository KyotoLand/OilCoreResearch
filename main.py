from sklearn.preprocessing import StandardScaler, MinMaxScaler
from cluster import get_kmeans_with_specified_metric
from quality import plot_coefficients_over_num_cl, plot_coefficients_over_num_cl_for_all_metrics, plot_clusters_umap, \
    plot_clusters_tsne, plot_cluster_regressions, plot_cluster_regressions_separate, cluster_quality_coefficients, \
    plot_cluster_regressions_separate_colored, plot_cluster_linear_regressions_separate_colored
from tools import read_xlsx_file, get_capillary_curves, get_porosity_and_permeability, save_clusters_pp_to_xlsx, \
    get_porosity, get_permeability, get_pp_by_fields_from_xlsx, get_values_from_excel, plot_capillary_curve, \
    get_capillary_curves_with_log_permeability, get_capillary_curves_with_terr
import numpy as np


distance_measures = {'euclidean': 0, 'squared euclidean': 1, 'manhattan': 2, 'chebyshev': 3,
                         'canberra': 5, 'chi-square': 6}


def get_fields_distribution():
    filename = 'data.xlsx'
    pp_by_fields = get_pp_by_fields_from_xlsx(filename)

    values = []
    labels = []

    for group_name, coords in pp_by_fields.items():
        for coord in coords:
            values.append(coord)
            labels.append(group_name)
    print(values)
    print(labels)
    print(len(values))
    print(len(labels))
    print(len(capillary_curves))
    print(list(capillary_curves.values())[0])
    print(list(capillary_curves.values())[-1])
    print(cluster_quality_coefficients(values, labels))

    column_index = 0
    start_row = 3
    value_map = {'терригенныф': 0, 'карбонатный': 1}
    collector_type = np.array(get_values_from_excel(filename, column_index, start_row, value_map))
    print(collector_type)

    plot_cluster_linear_regressions_separate_colored(np.array(values), labels, collector_type)


def get_clusters_distribution(clusters):
    data = read_xlsx_file('data.xlsx')
    x_raw = get_porosity_and_permeability(data)
    x = list(x_raw.values())
    # print(X)
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    values, labels = get_kmeans_with_specified_metric(x, 0, clusters)

    # print(values)
    # print(labels)
    # print(len(values))
    # print(len(labels))
    # print(len(capillary_curves))
    # print(list(capillary_curves.values())[0])
    # print(list(capillary_curves.values())[-1])
    # print(cluster_quality_coefficients(values, labels))

    column_index = 0
    start_row = 3
    value_map = {'терригенныф': 0, 'карбонатный': 1}
    collector_type = np.array(get_values_from_excel('data.xlsx', column_index, start_row, value_map))
    # print(collector_type)

    plot_cluster_linear_regressions_separate_colored(np.array(list(x_raw.values())), labels, collector_type)


if __name__ == '__main__':
    # define dictionary for distance measures
    data_matrix = read_xlsx_file('data.xlsx')
    capillary_curves = get_capillary_curves(data_matrix)
    pp_data = get_porosity_and_permeability(data_matrix)
    porosity = get_porosity(data_matrix)
    permeability = get_permeability(data_matrix)
    # print(permeability)
    # print(pp_data)

    pp_values = list(pp_data.values())
    capillary_curves_values = list(capillary_curves.values())
    # print(X)
    scaler = MinMaxScaler()
    capillary_curves_scaled = scaler.fit_transform(capillary_curves_values)
    pp_values_scaled = scaler.fit_transform(pp_values)
    porosity_values_scaled = scaler.fit_transform(list(porosity.values()))
    permeability_values_scaled = scaler.fit_transform(list(permeability.values()))

    # plot_coefficients_over_num_cl_for_all_metrics(pp_values_scaled, 20)
    x, labels = get_kmeans_with_specified_metric(pp_values_scaled, 2, 10)
    # save_clusters_to_xlsx(np.array(pp_values), labels, 'cluster.xlsx')
    # plot_clusters_tsne(np.array(values), labels)
    # plot_cluster_regressions(np.array(values), labels)
    # key = list(capillary_curves.keys())[650]
    # plot_capillary_curve(capillary_curves.get(key), key)
    # print(len(capillary_curves_values))
    print(len(labels))
    collector_type = np.zeros(len(labels))
    # plot_cluster_linear_regressions_separate_colored(np.array(capillary_curves_values), labels, collector_type)
    # get_fields_distribution()
    num_clusters = 4
    get_clusters_distribution(num_clusters)


