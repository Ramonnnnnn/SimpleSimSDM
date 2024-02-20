import os
import pandas as pd
import matplotlib.pyplot as plt

class MultiCurvePlotter:
    def __init__(self, directory, output_dir):
        self.directory = directory
        self.output_dir = output_dir
        self.data = {}

    def parse_csv_files(self):
        for filename in os.listdir(self.directory):
            if filename.endswith(".csv"):
                metric, algorithm = filename.split("_")[:2]
                metric = metric.capitalize()  # Capitalize metric name
                if metric not in self.data:
                    self.data[metric] = {}
                df = pd.read_csv(os.path.join(self.directory, filename))
                self.data[metric][algorithm] = df

    def plot_curves(self):
        line_styles = ['-', '--', '-.', ':']  # Different line styles
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Different colors
        for metric, curves in self.data.items():
            plt.figure(figsize=(10, 6))
            plt.title(f"{metric} vs Erlangs")
            for i, (algorithm, df) in enumerate(curves.items()):
                # Remove '.csv' from the legend
                algorithm = algorithm.replace(".csv", "")
                style = line_styles[i % len(line_styles)]
                color = colors[i % len(colors)]
                plt.errorbar(df['load'], df['value'], yerr=df['error'], label=algorithm, linestyle=style, color=color)
            plt.xlabel('Load')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            output_path = os.path.join(self.output_dir, f"{metric.lower()}_curves.png")
            plt.savefig(output_path)
            plt.close()


