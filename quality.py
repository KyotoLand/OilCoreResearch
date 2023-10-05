import lmfit as lmfit
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from cluster import get_kmeans_with_specified_metric


distance_measures = {'euclidean': 0, 'squared euclidean': 1, 'manhattan': 2, 'chebyshev': 3,
                         'canberra': 5, 'chi-square': 6}


def cluster_quality_coefficients(data, labels):
    # Silhouette coefficient
    s_score = silhouette_score(data, labels)

    # Calinski-Harabasz index
    ch_score = calinski_harabasz_score(data, labels)

    # Davies-Bouldin index
    db_score = davies_bouldin_score(data, labels)

    return s_score, ch_score, db_score


def plot_coefficients_over_num_cl(X, max_k, metric):
    """
    Функция для построения зависимости значений коэффициентов
     качества кластеризации от количества кластеров для данной метрики.

    :param metric: метрика для расчета расстояния
    :param X: массив данных для кластеризации
    :param max_k: максимальное количество кластеров для тестирования
    :return: оптимальное количество кластеров
    """

    # создание списка для хранения значений коэффициента Дэвиса-Болдина для каждого количества кластеров
    db_scores = []
    ch_scores = []
    s_scores = []

    # тестирование количества кластеров от 2 до max_k
    for k in range(2, max_k + 1):
        # создание модели k-means
        x, labels = get_kmeans_with_specified_metric(X,metric, k)
        s_score, ch_score, db_score = cluster_quality_coefficients(x, labels)
        db_scores.append(db_score)
        ch_scores.append(ch_score)
        s_scores.append(s_score)

    # построение графиков зависимости коэффициентов от количества кластеров
    fig, axs = plt.subplots(3, 1, figsize=(4, 6))
    clusters = range(2, max_k + 1)
    # Создаем первый график
    axs[0].plot(clusters, db_scores)
    axs[0].set_title('Davies-Bouldin index')
    axs[0].set_xlabel('Number of clusters')
    axs[0].set_ylabel('Davies-Bouldin index')

    # Создаем второй график
    axs[1].plot(clusters, ch_scores)
    axs[1].set_title('Calinski-Harabasz index')
    axs[1].set_xlabel('Number of clusters')
    axs[1].set_ylabel('Calinski-Harabasz index')

    # Создаем третий график
    axs[2].plot(clusters, s_scores)
    axs[2].set_title('Silhouette coefficient')
    axs[2].set_xlabel('Number of clusters')
    axs[2].set_ylabel('Silhouette coefficient')
    fig.subplots_adjust(hspace=1)
    # Отображаем графики
    plt.grid()
    plt.show()


def calculate_k_nearest_neighbours_dist(X, k):

    # Compute the k-distance graph
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    kth_distances = distances[:, -1]

    # Sort the kth distances and plot them
    kth_distances_sorted = np.sort(kth_distances)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(kth_distances_sorted)
    ax.set_xlabel('Points')
    ax.set_ylabel(f'{k}-th distance')
    plt.savefig(f'{k}-th distance.png')
    plt.show()


def plot_clusters_umap(X, labels):
    # Perform t-SNE for visualization
    umap_ = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_.fit_transform(X)
    # Plot the clusters
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='rainbow')
    legend = ax.legend(*scatter.legend_elements(), title='Clusters')
    ax.add_artist(legend)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    plt.show()


def plot_clusters_tsne(X, labels):
    if len(X[0]) > 2:
        # Perform t-SNE dimensionality reduction on the data
        tsne = TSNE(n_components=2, random_state=0)
        tsne_values = tsne.fit_transform(X)
    else:
        tsne_values = X
    # Get the unique labels
    unique_labels = np.unique(labels)

    # Plot the clusters
    plt.figure(figsize=(8, 8))
    for label in unique_labels:
        # Get the indices of the data points with this label
        label = str(label)
        labels = [str(label) for label in labels]
        indices = np.where(np.char.equal(labels, label))[0]
        # Plot the data points with this label
        plt.scatter(tsne_values[indices, 1], tsne_values[indices, 0], label=label)
    # Add a legend and show the plot
    plt.legend()
    plt.yscale('log')
    plt.show()


def plot_coefficients_over_num_cl_for_all_metrics(X, max_k):
    """
    Функция для построения зависимости значений коэффициентов
     качества кластеризации от количества кластеров для данной метрики.

    :param metric: метрика для расчета расстояния
    :param X: массив данных для кластеризации
    :param max_k: максимальное количество кластеров для тестирования
    :return: оптимальное количество кластеров
    """

    # создание списка для хранения значений коэффициента Дэвиса-Болдина для каждого количества кластеров
    db_scores = {}
    ch_scores = {}
    s_scores = {}

    # тестирование количества кластеров от 2 до max_k
    for measure, value in distance_measures.items():
        db_scores.update({measure: []})
        ch_scores.update({measure: []})
        s_scores.update({measure: []})
        for k in range(2, max_k + 1):
            # создание модели k-means
            x, labels = get_kmeans_with_specified_metric(X, value, k)
            s_score, ch_score, db_score = cluster_quality_coefficients(x, labels)
            db_scores.get(measure).append(db_score)
            ch_scores.get(measure).append(ch_score)
            s_scores.get(measure).append(s_score)

    # построение графиков зависимости коэффициентов от количества кластеров
    fig, axs = plt.subplots(3, 6, figsize=(15, 8))
    columns = list(distance_measures.keys())
    for i in range(len(columns)):
        axs[0, i].set_title(columns[i] + " metric" +
                            " \n DB index = " +
                            str(round(sum(db_scores.get(columns[i]))/len(db_scores.get(columns[i])), 2)) +
                            "\n CH index = " +
                            str(round(sum(ch_scores.get(columns[i]))/len(db_scores.get(columns[i])), 2)) +
                            "\n Silhouette index = " +
                            str(round(sum(s_scores.get(columns[i]))/len(db_scores.get(columns[i])), 2)), fontsize=10)

    # Добавляем названия графиков
    axs[0, 0].set_ylabel('Davies-Bouldin index', fontsize=10)
    axs[1, 0].set_ylabel('Calinski-Harabasz index', fontsize=10)
    axs[2, 0].set_ylabel('Silhouette coefficient', fontsize=10)
    clusters = range(2, max_k + 1)
    for i in range(6):
        axs[2, i].set_xlabel('Number of clusters', fontsize=12)
        axs[0, i].plot(clusters, db_scores.get(columns[i]))
        axs[0, i].grid(True)
        axs[1, i].plot(clusters, ch_scores.get(columns[i]))
        axs[1, i].grid(True)
        axs[2, i].plot(clusters, s_scores.get(columns[i]))
        axs[2, i].grid(True)
        # axs[0, i].set_xlabel('Number of clusters', fontsize=12)
        # axs[1, i].set_xlabel('Number of clusters', fontsize=12)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.suptitle('Коэффициенты качества кластеризации для разных метрик. MinMaxScaling по проницаемости.', fontsize=14)
    # Отображаем графики
    plt.show()


def plot_cluster_regressions(values, labels):
    # Get unique labels
    unique_labels = np.unique(labels)

    # Set up plot
    fig, ax = plt.subplots()

    # Loop through unique labels
    for label in unique_labels:
        # Get values for current label
        label = str(label)
        labels = [str(label) for label in labels]
        label_values = values[np.where(np.char.equal(labels, label))[0]]
        label_values = label_values[np.argsort(label_values[:, 1])]
        x = label_values[:, 1]
        y = label_values[:, 0]
        y_mean = np.mean(y)
        mask = y <= 5 * y_mean
        # Calculate regression between first and second value
        # x = x[mask]
        # y = y[mask]

        # slope, intercept = np.polyfit(x, y, 1)

        # Calculate MSE and mean x and y in group
        #y_pred = slope * x + intercept

        # Fit an exponential function
        from scipy.optimize import curve_fit

        def exponential_func(x, a, b):
            return a * np.exp(-b * x)

        popt, pcov = curve_fit(exponential_func, x, y, p0=(0.6, -0.3), maxfev=2000)

        y_pred = exponential_func(x, *popt)
        mse_perm = np.sqrt(np.std(y))
        mse_poro = np.sqrt(np.std(x))
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        # Plot regression line with legend
        ax.plot(x, y_pred, label=f'Cluster {label}: Permeability={mean_y:.3f} +- {mse_perm:.3f} mD, Porosity={mean_x:.3f} +- {mse_poro:.3f} %, R-square={r2:.3f}')

    # Add legend and labels
    ax.legend(fontsize= 8)
    ax.set_xlabel('Porosity')
    ax.set_ylabel('Permeability')
    ax.set_title('Exponential Regression in Clusters')
    plt.yscale('log')
    plt.show()


def plot_cluster_regressions_separate(values, labels):
    # Get unique labels
    unique_labels = np.unique(labels)

    # Set up plot
    n_cols = 3
    n_rows = int(np.ceil(len(unique_labels) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axs = axs.flatten()

    # Loop through unique labels
    for i, label in enumerate(unique_labels):
        # Get values for current label
        label = str(label)
        labels = [str(label) for label in labels]
        label_values = values[np.where(np.char.equal(labels, label))[0]]

        label_values = label_values[np.argsort(label_values[:, 1])]
        # Exclude y points that are more than 50 times the mean of y
        x = label_values[:, 1]
        y = label_values[:, 0]
        y_mean = np.mean(y)
        mask = y <= 5 * y_mean
        # x = x[mask]
        # y = y[mask]

        # Fit an exponential function
        def exponential_func(x, a, b):
            return a * np.exp(-b * x)

        mod = lmfit.Model(exponential_func)

        # Estimate initial parameters using guess()
        params = mod.make_params(a=0.1, b=-0.5)
        params['a'].set(min=0.01, max=0.6)
        params['b'].set(min=-0.8, max=-0.1)
        result = mod.fit(y, params, x=x)

        popt = [result.best_values['a'], result.best_values['b']]
        pcov = result.covar

        y_pred = exponential_func(x, *popt)
        mse_perm = np.sqrt(np.std(y))
        mse_poro = np.sqrt(np.std(x))
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Plot regression line with legend
        ax = axs[i]
        ax.plot(x, y_pred, label=f'Cluster {label}:\n Porosity={mean_x:.3f} +- {mse_poro:.3f} %,\n Permeability={mean_y:.3f} +- {mse_perm:.3f} mD,\n R-square={r2:.3f},\n Exponential fit')
        ax.scatter(x, y, alpha=0.5)

        # Add legend and labels
        ax.legend(fontsize= 6)
        ax.set_xlabel('Porosity')
        ax.set_ylabel('Permeability')
        ax.set_title(f'Cluster {label}')
        ax.set_yscale('log')
    # Remove unused subplots
    for j in range(len(unique_labels), n_rows * n_cols):
        fig.delaxes(axs[j])
    fig.subplots_adjust(hspace=1.8)
    plt.tight_layout()
    plt.show()


def plot_cluster_regressions_separate_colored(values, labels, colors):
    # Get unique labels
    unique_labels = np.unique(labels)

    # Set up plot
    n_cols = 3
    n_rows = int(np.ceil(len(unique_labels) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axs = axs.flatten()

    # Loop through unique labels
    for i, label in enumerate(unique_labels):
        # Get values for current label
        label = str(label)
        labels = [str(label) for label in labels]
        indicies = np.where(np.char.equal(labels, label))[0]
        label_values = values[indicies]
        color_values = colors[indicies]  # new line to select corresponding colors

        label_values = label_values[np.argsort(label_values[:, 1])]
        # Exclude y points that are more than 50 times the mean of y
        x = label_values[:, 1]
        y = label_values[:, 0]
        y_mean = np.mean(y)
        mask = y <= 5 * y_mean

        # Fit an exponential function
        def exponential_func(x, a, b):
            return a * np.exp(-b * x)

        mod = lmfit.Model(exponential_func)

        # Estimate initial parameters using guess()
        params = mod.make_params(a=0.2, b=-0.5)
        # params['a'].set(min=0.01, max=0.6)
        # params['b'].set(min=-0.8, max=-0.1)
        result = mod.fit(y, params, x=x)

        popt = [result.best_values['a'], result.best_values['b']]
        pcov = result.covar

        y_pred = exponential_func(x, *popt)
        mse_perm = np.sqrt(np.std(y))
        mse_poro = np.sqrt(np.std(x))
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Plot regression line with legend
        ax = axs[i]
        ax.plot(x, y_pred, label=f'Cluster {label}:\n Porosity={mean_x:.3f} +- {mse_poro:.3f} %,\n Permeability={mean_y:.3f} +- {mse_perm:.3f} mD,\n R-square={r2:.3f},\n Exponential fit')
        # Create a color map with yellow for 0 and purple for 1
        cmap = {0: 'salmon', 1: 'navy'}
        colors_tp = [cmap[color] for color in color_values]

        # Plot the scatter plot with the new colors
        ax.scatter(x, y, c=colors_tp, alpha=0.5)

        # Add legend and labels
        ax.legend(fontsize=6)
        ax.set_xlabel('Porosity')
        ax.set_ylabel('Permeability')
        ax.set_title(f'Cluster {label}')
        ax.set_yscale('log')
    # Remove unused subplots
    for j in range(len(unique_labels), n_rows * n_cols):
        fig.delaxes(axs[j])
    fig.subplots_adjust(hspace=1.8)
    plt.tight_layout()
    plt.show()


def plot_cluster_linear_regressions_separate_colored(values, labels, colors):
    # Get unique labels
    unique_labels = np.unique(labels)

    # Set up plot
    n_cols = 3
    n_rows = int(np.ceil(len(unique_labels) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    axs = axs.flatten()

    # Loop through unique labels
    for i, label in enumerate(unique_labels):
        # Get values for current label
        label = str(label)
        labels = [str(label) for label in labels]
        indicies = np.where(np.char.equal(labels, label))[0]
        label_values = values[indicies]
        color_values = colors[indicies]  # new line to select corresponding colors

        label_values = label_values[np.argsort(label_values[:, 1])]
        # Exclude y points that are more than 50 times the mean of y
        x = label_values[:, 1]
        y = np.log10(label_values[:, 0])
        y_mean = np.mean(y)
        print(y)
        mask = y <= 5 * y_mean

        # Fit an exponential function
        def linear_func(x, a, b):
            return a * x + b

        mod = lmfit.Model(linear_func)

        # Estimate initial parameters using guess()
        params = mod.make_params(a=10, b=5)
        # params['a'].set(min=0.01, max=0.6)
        # params['b'].set(min=-0.8, max=-0.1)
        result = mod.fit(y, params, x=x)

        popt = [result.best_values['a'], result.best_values['b']]
        pcov = result.covar

        y_pred = linear_func(x, *popt)
        mse_perm = np.sqrt(np.std(10**y))
        mse_poro = np.sqrt(np.std(x))
        mean_x = np.mean(x)
        mean_y = np.mean(10**y)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Plot regression line with legend
        ax = axs[i]
        ax.plot(x, 10**y_pred, label=f'Cluster {label}:\n Porosity={mean_x:.3f} +- {mse_poro:.3f} %,\n Permeability={mean_y:.3f} +- {mse_perm:.3f} mD,\n R-square={r2:.3f},\n Exponential fit')
        # Create a color map with yellow for 0 and purple for 1
        cmap = {0: 'salmon', 1: 'navy'}
        colors_tp = [cmap[color] for color in color_values]

        # Plot the scatter plot with the new colors
        ax.scatter(x, 10**y, c=colors_tp, alpha=0.5)

        # Add legend and labels
        ax.legend(fontsize=8)
        ax.set_xlabel('Porosity, %')
        ax.set_ylabel('Permeability, mD')
        ax.set_title(f'Cluster {label}')
        ax.set_yscale('log')
        ax.grid(True)
    # Remove unused subplots
    for j in range(len(unique_labels), n_rows * n_cols):
        fig.delaxes(axs[j])
    fig.subplots_adjust(hspace=2)
    plt.tight_layout()

    plt.show()

