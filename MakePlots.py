import FinalStatisticsPlotter

# Instantiate Plotter
grapher = FinalStatisticsPlotter.MultiCurvePlotter("C:\\Users\\ramon\\Downloads\\SimpleSimSDM\\SimpleSimSDM\\CVSs", "C:\\Users\\ramon\\Downloads\\SimpleSimSDM\\SimpleSimSDM\\plots")
# Plot metrics
grapher.parse_csv_files()
grapher.plot_curves()