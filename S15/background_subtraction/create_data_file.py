import os
import csv

root = "D:/Projects/theschoolofai/datasets/background_subtraction/valdata/data/"

file_list = os.listdir(root)
print(len(file_list),"images")

def write_list_to_file(file_list, filename):
    """Write the list to csv file."""

    with open(filename, "w") as outfile:
        outfile.write("filename")
        outfile.write("\n")
        for entries in file_list:
            outfile.write(entries)
            outfile.write("\n")

write_directory = root+"../output.csv"

write_list_to_file(file_list, write_directory)