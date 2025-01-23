import os
import pickle
import numpy as np
import csv
class Helper:
    def __init__(self):
        pass
    @staticmethod
    def load(parameter_path):
        """Load model parameters from a file."""
        if not os.path.exists(parameter_path):
            raise FileNotFoundError(f"Parameter file not found: {parameter_path}")

        with open(parameter_path, "rb") as f:
            parameters = pickle.load(f)

        if not isinstance(parameters, np.ndarray) or parameters.dtype != object:
            raise ValueError("Loaded parameters are not in the expected numpy ndarray format with dtype=object.")
    
        return [param for param in parameters]
    @staticmethod
    def save(parameters_np, parameter_path):
        """Save model parameters to a file."""
        if not isinstance(parameters_np, (np.ndarray, list)):
            raise ValueError("Parameters must be a numpy ndarray or a list of numpy arrays.")
    
        if isinstance(parameters_np, list):
            # Convert list of NumPy arrays to a single numpy.ndarray with dtype=object
            parameters_np = np.array(parameters_np, dtype=object)

        os.makedirs(os.path.dirname(parameter_path), exist_ok=True)

        with open(parameter_path, "wb") as f:
            pickle.dump(parameters_np, f)

def get_helper() -> Helper:
    """Return an instance of the Helper class."""
    return Helper()
def ReadWriteFile():
    input_file = 'CentralServer1000epoch/train/results.csv'  # Replace with your actual CSV file path

    # Output text file path
    output_file = 'extracted_columns.txt'  # Replace with your desired output file name

    # Open the CSV file for reading and the text file for writing
    with open(input_file, mode='r') as csv_file, open(output_file, mode='w') as text_file:
        csv_reader = csv.reader(csv_file)  # Initialize CSV reader
        for row in csv_reader:
            # Extract columns 0, 6, and 8 (0-based index)
            # Ensure there are enough columns in the row to avoid IndexError
            if len(row) > max(0, 6, 7):
                col_0 = row[0].strip()
                col_6 = row[6].strip()
                col_8 = row[7].strip()
                # Write the extracted columns to the text file
                text_file.write(f"{col_0},{col_6},{col_8}\n")

    print(f"Extracted data saved to {output_file}")    