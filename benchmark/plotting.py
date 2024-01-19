import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    return pd.read_csv(file_path)

def plot_graphs(df):
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input is not a valid DataFrame")

    if df.isnull().any().any():
        raise ValueError("DataFrame contains missing values")

    metrics = ['time', 'hausdorff_distance', 'chamfer_distance', 'curvature_error', 'memory']
    fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(10, 5 * len(metrics)))

    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame")

        sns.lineplot(x='simplification_rate', y=metric, hue='index_simplification_method', data=df, ax=axes[i], marker='o')
        axes[i].set_title(f'{metric.title()} vs Simplification Rate')
        axes[i].set_xlabel('Simplification Rate')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].legend(title='Simplification Method')

    plt.tight_layout()
    plt.show()
