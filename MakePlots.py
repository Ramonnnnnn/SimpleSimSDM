import FinalStatisticsPlotter

# Instantiate Plotter
grapher = FinalStatisticsPlotter.MultiCurvePlotter("/Users/ramonoliveira/Desktop/CG - Relatório 3D/SimpleSIm/CVSs", "/Users/ramonoliveira/Desktop/CG - Relatório 3D/SimpleSIm/plots")
# Plot metrics
grapher.parse_csv_files()
grapher.plot_curves()