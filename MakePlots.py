import FinalStatisticsPlotter

# Instantiate Plotter
# grapher = FinalStatisticsPlotter.MultiCurvePlotter("C:\\Users\\ramon\\Downloads\\SimpleSimSDM\\SimpleSimSDM\\CSVs", "C:\\Users\\ramon\\Downloads\\SimpleSimSDM\\SimpleSimSDM\\plots")
grapher = FinalStatisticsPlotter.MultiCurvePlotter("C:\\Users\\ramon\\Downloads\\SimpleSimSDM\\SimpleSimSDM\\CSV_final_comparison\\PAN", "C:\\Users\\ramon\\Downloads\\SimpleSimSDM\\SimpleSimSDM\\CSV_final_comparison\\PAN")

# Plot metrics
grapher.parse_csv_files()
grapher.plot_curves()