import csv


def print_columns_as_arrays(filename):
    # Initialize empty lists for each column
    load = []
    value = []
    error = []

    # Open the CSV file
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)

        # Loop through each row in the CSV
        for row in csv_reader:
            load.append(int(row['load']))
            value.append(float(row['value']))
            error.append(float(row['error']))

    # Print the arrays
    print("load =", load)
    print("value =", value)
    print("error =", error)


# Replace 'your_file.csv' with your actual file name
filename = 'crosstalk_fca(USA.newest).csv'
print_columns_as_arrays(filename)
