import os
import torch

def load_embedding_files(base_dir="output/missing_aware"):
    """
    Loads only the `embedding_before_classifier.pt` and `embedding_after_classifier.pt` files
    from the specified directory structure.
    
    Parameters:
      - base_dir (str): The base directory where the embedding files are stored.
      
    Returns:
      - embeddings (dict): A dictionary containing loaded embeddings organized by path.
    """
    processed_embeddings = {'before_classifier': {}, 'after_classifier': {}}
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file in ["embedding_before_classifier.pt", "embedding_after_classifier.pt"]:
                    file_path = os.path.join(root, file)
                    
                    # Load embedding tensor
                    embeddings = torch.load(file_path)  # Shape: [num_batches, batch_size, embedding_size]
                    
                    # Remove padding rows (rows with all zeros)
                    filtered_embeddings = []
                    for batch in embeddings:
                        # Keep only rows that are not all zeros
                        non_zero_rows = batch[~torch.all(batch == 0, dim=1)]
                        filtered_embeddings.append(non_zero_rows)
                    
                    # Concatenate filtered embeddings across batches
                    processed_embedding = torch.cat(filtered_embeddings, dim=0)  # Shape: [total_samples, embedding_size]
                    
                    # Store in processed_embeddings dict
                    if file == "embedding_before_classifier.pt":
                        processed_embeddings['before_classifier'] = processed_embedding
                    elif file == "embedding_after_classifier.pt":
                        processed_embeddings['after_classifier'] = processed_embedding
                    
                    print(f"Processed and reshaped embedding from {file_path} to {processed_embedding.shape}")
        
    return processed_embeddings

# Example usage
embedding_data = load_embedding_files()
print(embedding_data['after_classifier'].shape, embedding_data['before_classifier'].shape)  # batch, 
torch.save(embedding_data['before_classifier'], "output/missing_aware/before_classifier.pt")
torch.save(embedding_data['after_classifier'], "output/missing_aware/after_classifier.pt")

# import pdb; pdb.set_trace()