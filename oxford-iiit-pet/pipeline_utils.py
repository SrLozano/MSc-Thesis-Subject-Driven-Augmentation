# This file contains some functions used in the main.py file.
# The functions are utils used to manipulate the data pipeline.

# Import dependencies
import re
import os
import shutil
import random

def get_breeds(filepath):
    """
    This function parses the samples, class, species and breed_id by breed from the file and it adds the information to a dictionary.
    :param filepath: The path to the file to parse the information from.
    """

    # Dictionary with breed as key and list of samples as value
    samples_by_breed = {}

    # Dictionaries with class, specie and breed as key and breed id as value
    class_by_id = {}
    species_by_id = {}
    breed_by_id = {}

    # Parse the breed and breed id from the file name and add id to the dictionary
    with open(filepath, 'r') as file:
        data = file.read()
        for row in data.split('\n'):
            if len(row) != 0:
                sample = row.split(' ')
                breed = re.sub(r'_\d+', '', sample[0])

                # If breed not in dictionary, add it. Else, append sample to list
                if breed not in samples_by_breed:
                    samples_by_breed[breed] = []
                else:
                    samples_by_breed[breed].append(sample[0])

                # If class not in dictionary, add it. 
                if breed not in class_by_id:
                    class_by_id[breed] = sample[1]
                
                # If specie not in dictionary, add it. 
                if breed not in species_by_id:
                    species_by_id[breed] = sample[2]

                # If breed not in dictionary, add it. 
                if breed not in breed_by_id:
                    breed_by_id[breed] = sample[3]

    return samples_by_breed, class_by_id, species_by_id, breed_by_id

def delete_files(dst, verbose=False):
    """
    This function deletes all files in a directory.
    :param dst: The directory to delete all files from.
    :param verbose: Indicates whether to print the information about the deleted files.
    """
    if len(os.listdir(dst)) > 0:
        for filename in os.listdir(dst):
            file_path = os.path.join(dst, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    shutil.rmtree(file_path)
            except Exception as e:
                if verbose:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

def remove_EOL_from_file(file_name):
    """
    This function removes the last newline character from a file.
    :param file_name: The file to remove the last newline character from.
    """
    with open(file_name, "r") as f:
        string_with_newline = f.read()

    string_without_last_newline = string_with_newline[:-1]

    with open(file_name, "w") as f:
        f.write(string_without_last_newline)

def create_splits(path):
    """
    This function creates the validation and test splits for the Oxford-IIIT Pet dataset.
    :param path: The path to the Oxford-IIIT Pet dataset.
    """

    # Location of the test file
    test = f'{path}/annotations/test.txt'

    # Get samples by breed, class, specie and breed id from txt file
    samples_by_breed, class_by_id, species_by_id, breed_by_id = get_breeds(test)

    # Create empty file for validation. test == validation
    with open('test.txt', 'w') as file:
        file.write('')

    # Create empty file for test 
    with open('final_test.txt', 'w') as file:
        file.write('')

    # Create validation and test splits. final_test == test
    for breed in samples_by_breed:

        # Shuffle the list into two splits for validation and test
        my_list = samples_by_breed[breed]
        random.Random(41).shuffle(my_list)
        half = len(my_list) // 2
        validation = my_list[:half]
        test = my_list[half:]

        # Write the validation split to the validation file. test == validation
        for element in validation:
            with open('test.txt', 'a') as file:
                file.write(f'{element} {class_by_id[breed]} {species_by_id[breed]}, {breed_by_id[breed]}\n')
        
        # Write the test split to the test file. final_test == test
        for element in test:
            with open('final_test.txt', 'a') as file:
                file.write(f'{element} {class_by_id[breed]} {species_by_id[breed]}, {breed_by_id[breed]}\n')

    # Remove the last newline character from the files
    remove_EOL_from_file('test.txt')
    remove_EOL_from_file('final_test.txt')

    # Move the files to the correct location
    shutil.move("test.txt", f'{path}/annotations/test.txt')
    shutil.move("final_test.txt", f'{path}/annotations/final_test.txt')