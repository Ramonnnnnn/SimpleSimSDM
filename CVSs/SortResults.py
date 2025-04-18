import os
import csv


def sort_csv_by_first_column(csv_folder):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder, csv_file)

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Read the header
            sorted_rows = sorted(reader, key=lambda row: float(row[0]))  # Sort rows by first column

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # Write the header back
            writer.writerows(sorted_rows)  # Write the sorted rows

        print(f"Sorted {csv_file}")


# Set the folder containing the CSV files
csv_folder = os.path.join(os.path.dirname(__file__), "")
sort_csv_by_first_column(csv_folder)
