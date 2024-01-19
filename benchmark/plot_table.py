import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter

def load_data(file_path):
    return pd.read_csv(file_path)

def group_and_sort(df, group_column, sort_column):
    df[group_column] = df[group_column].round(1)
    df = df.sort_values([group_column, sort_column], ascending=[True, True])
    return df

def plot_table(df, highlight_columns=None):
    if df.empty:
        print("DataFrame is empty. Skipping table creation.")
        return

    highlight_color = ColorConverter().to_rgba('yellow', alpha=0.3)

    fig, ax = plt.subplots(figsize=(12, len(df) * 0.3))

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)

    if highlight_columns:
        for i, col in enumerate(df.columns):
            if col in highlight_columns:
                for j, cell in enumerate(table.get_celld().values()):
                    if cell.get_text().get_text() == col:
                        cell.set_facecolor(highlight_color)

    plt.tight_layout()

    plt.savefig('table_plot.png')
    plt.show()

if __name__ == "__main__":
    file_path = 'benchmark/data/output/csv/metrics_results.csv'
    filech = 'benchmark/data/output/csv/measures.csv'
    data = load_data(file_path)
    dataa = load_data(filech)
    grouped_data = group_and_sort(data, 'simplification_rate', 'simplification_rate')
    groupeddata = group_and_sort(dataa, 'simplification_rate', 'simplification_rate')

    highlight_cols = ['simplification_rate', 'simplification_rate']
    plot_table(grouped_data, highlight_columns=highlight_cols)
    plot_table(groupeddata, highlight_columns=highlight_cols)