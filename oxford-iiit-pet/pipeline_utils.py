# Import dependencies
import re

def get_breeds(filepath):

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
