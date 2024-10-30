import torch
import os

def load_batch_data():
    file_path = f"output/imdb/L2P_Prob/train/sample_data.pt"
    
    if not os.path.exists(file_path):
        print(f"No data file found")
        return None
    
    # Load the data
    batch_data = torch.load(file_path)
    
    # Verify contents
    print(f"Loaded data:")
    print(f"Number of batches stored: {len(batch_data)}\n")
    
    for i, data in enumerate(batch_data):
        print(f"Batch {i + 1}:")
        print(f"  Local Prompts Shape: {data['local_prompts'].shape}")
        print(f"  Summarizing Prompts Shape: {data['summarizing_prompts'].shape}")
        print(f"  Missing Type: {data['missing_type']}")
        print(f"  Embedding Before Classifier Shape: {data['embedding_before_classifier'].shape}")
        print(f"  Embedding After Classifier Shape: {data['embedding_after_classifier'].shape}\n")
    
    return batch_data

# Example usage
 # Replace with the desired batch_id to load
batch_data = load_batch_data()
import pdb; pdb.set_trace()
