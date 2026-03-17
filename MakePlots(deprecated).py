import FinalStatisticsPlotter

# This function will plot every csv in a given directory directly to the output directory

# Instantiate Plotter
grapher = FinalStatisticsPlotter.MultiCurvePlotter("CSVs/salao_test", "plots")

# Plot metrics
grapher.parse_csv_files()
grapher.plot_curves()