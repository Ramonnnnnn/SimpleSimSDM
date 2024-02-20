import csv
import os

class Logger:
    def __init__(self, folder_path, csv_files):
        self.folder_path = folder_path
        self.csv_files = csv_files
        self.file_handles = {}

        # Create CSV files if they don't exist
        for file_name in self.csv_files:
            file_path = os.path.join(self.folder_path, file_name)
            if not os.path.exists(file_path):
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['load', 'value', 'error'])

    def add_datapoint(self, csv_file, load, value, error):
        if csv_file not in self.csv_files:
            print(f"Error: {csv_file} is not registered.")
            return

        file_path = os.path.join(self.folder_path, csv_file)
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([load, value, error])
