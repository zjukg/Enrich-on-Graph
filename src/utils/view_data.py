
from datasets import load_dataset
import random

def write_to_file(top_k_samples, file_name="view_data_result.txt"):
    """
    Write the first k samples to a file.
    :param top_k_samples: list, containing dictionaries of the first k samples
    :param file_name: str, path of the file to write to
    """
    # Convert dictionary content to string format (e.g., JSON format)
    import json
    top_k_samples_str = json.dumps(top_k_samples, indent=4, ensure_ascii=False)
    
    # Open file and overwrite
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(top_k_samples_str)

# Load dataset
dataset = load_dataset(
    "parquet", 
    data_files={'test': 'preprocess_datasets/qa_datasets/cwq_gpt-4o-mini-2024-07-18_eog_qa_2025-09-06_19-38-15.parquet'}
)

# Access test set
test_dataset = dataset['test']

print("This dataset information:", test_dataset)

# Access the first top_k samples
top_k = 10
top_k_samples = [test_dataset[i] for i in range(top_k)]

write_path = "view_data_result.txt"

# Print key names of the first k samples (only print key names of the first sample as reference)
for key in top_k_samples[0].keys():
    print(key)

# Call function to write top_k_samples to file
write_to_file(top_k_samples, write_path)
