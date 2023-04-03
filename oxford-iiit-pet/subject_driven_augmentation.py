# Import dependencies
import os
import time
import torch
import shutil
import pipeline_utils 


# Define parameters
number_of_samples = 5
subject_driven_technique = "dreambooth"

path_to_dataset = "../../../../../../work3/s226536/datasets/oxford-iiit-pet"


# Create folder for saved models
saved_models_path = f"../../../../../../work3/s226536/saved_models/{subject_driven_technique}-{number_of_samples}"
os.makedirs(saved_models_path, exist_ok=True)

# Get samples by breed, class, specie and breed id from txt file
samples_by_breed, class_by_id, species_by_id, breed_by_id = pipeline_utils.get_breeds(f'{path_to_dataset}/annotations/trainval.txt')

breeds_to_generate = list(samples_by_breed.keys())[0:2]
#breeds_to_generate = ['Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

# Data augmentation generation for the selected breeds
for breed in breeds_to_generate:

    print(f"-------------------------------------\nStarted subject-driven fine tunning for breed {breed}...\n")
    start_time = time.time()
    
    # Define destination for subject-driven generation algorithm
    dst = f'/zhome/d1/6/191852/MSc-thesis/experiments/03-oxford-iiit-pet/dataset'

    # Delete all files in dst
    pipeline_utils.delete_files(dst)

    # Copy selected random samples to dst to use them in the subject-driven generation. Samples already shuffled by prepare_dataset.py 
    for sample_image in samples_by_breed[breed][:number_of_samples]:
        src = f'{path_to_dataset}/images/{sample_image}.jpg'
        shutil.copy(src, dst)

    print(f"Executing subject-driven technique...\n")

    # Call subject-driven generation algorithm
    if subject_driven_technique == "textual-inversion": 
        os.system('accelerate launch /zhome/d1/6/191852/MSc-thesis/textual-inversion/huggingFace/textual_inversion.py \
            --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
            --train_data_dir="/zhome/d1/6/191852/MSc-thesis/experiments/03-oxford-iiit-pet/dataset" \
            --learnable_property="object" \
            --placeholder_token="<funny-ret>" --initializer_token="animal" \
            --resolution=512 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=4 \
            --max_train_steps=3000 \
            --learning_rate=5.0e-04 --scale_lr \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --output_dir="/zhome/d1/6/191852/saved_model" \
            ') 
    elif subject_driven_technique == "dreambooth":
        os.system('accelerate launch /zhome/d1/6/191852/MSc-thesis/dreambooth/train_dreambooth.py \
            --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
            --instance_data_dir="/zhome/d1/6/191852/MSc-thesis/experiments/03-oxford-iiit-pet/dataset" \
            --output_dir="/zhome/d1/6/191852/saved_model" \
            --instance_prompt="<funny-ret>" \
            --resolution=512 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=1 \
            --learning_rate=5e-6 \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --max_train_steps=400 \
            ')
    
    # Move saved_model folder to scratch space and create new saved_model folder
    shutil.move("/zhome/d1/6/191852/saved_model", saved_models_path + f"/{breed}")
    os.mkdir("/zhome/d1/6/191852/saved_model")

    # Time elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds\n")

    # Clear memory
    torch.cuda.empty_cache()