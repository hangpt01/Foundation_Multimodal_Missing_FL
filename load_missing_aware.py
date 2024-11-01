import torch
import os

# or imdb
dataset = "food101"
# or missing_aware
model = "missing_aware"
flag = "test"

def load_all_saved_data(flag, client_id=None, current_round=None):
    """
    Loads and inspects saved data files for a given flag, client, and round.
    
    Parameters:
      - flag (str): Indicates whether to load 'train' or 'test' data.
      - client_id (int, optional): If provided, loads data for a specific client.
      - current_round (int, optional): If provided, loads data for a specific round.
      
    Returns:
      - all_data (dict): A dictionary with keys as file paths and values as loaded data.
    """
    base_path = "output/{}/{}/".format(dataset, model)
    all_data = {}
    
    if flag == "train":
        train_path = os.path.join(base_path, "train")
        # Iterate over clients if client_id is not specified
        clients = [f"client_{client_id}"] if client_id else os.listdir(train_path)
        
        for client_folder in clients:
            client_path = os.path.join(train_path, client_folder)
            if not os.path.isdir(client_path):
                continue

            # Load data files for specified or all rounds
            rounds = [f"sample_data_round_{current_round}.pt"] if current_round else os.listdir(client_path)
            for file_name in rounds:
                file_path = os.path.join(client_path, file_name)
                if os.path.exists(file_path):
                    data = torch.load(file_path)
                    all_data[file_path] = data
                    print(f"Loaded data from {file_path}:")
                    # display_data(data)
    
    elif flag == "test":
        test_path = os.path.join(base_path, "test")
        rounds = [f"sample_data_round_{current_round}.pt"] if current_round else os.listdir(test_path)
        
        for file_name in rounds:
            file_path = os.path.join(test_path, file_name)
            if os.path.exists(file_path):
                data = torch.load(file_path)
                all_data[file_path] = data
                print(f"Loaded data from {file_path}:")
                # display_data(data)

    else:
        print(f"Invalid flag: {flag}. Use 'train' or 'test'.")
    
    return all_data

def display_data(data):
    """
    Helper function to display data in a human-readable format.
    """
    # import pdb; pdb.set_trace()
    print(f"Number of batches: {len(data)}")
    for i, batch_data in enumerate(data):
        print(f"\nBatch {i + 1}:")
        # import pdb; pdb.set_trace()
        if model == "L2P_Prob":
            print(i, f"  Local Prompts Shape: {batch_data['local_prompts'].shape}")
            print(i, f"  Summarizing Prompts Shape: {batch_data['summarizing_prompts'].shape}")
        else:
            print(i, f"  Prompts Shape: {batch_data['prompts'].shape}")
        
        # print(f"  Summarizing Prompts Shape: {batch_data['prompts'].shape}")
        print(f"  Missing Type: {batch_data['missing_type']}")
        print(f"  Embedding Before Classifier Shape: {batch_data['embedding_before_classifier'].shape}")
        print(f"  Embedding After Classifier Shape: {batch_data['embedding_after_classifier'].shape}")

def pad_embeddings(embeddings, max_len):
    """
    Pads a list of embeddings to the specified max length along dimension 0.
    
    Parameters:
      - embeddings (list of Tensors): List of 2D tensors of varying first dimensions.
      - max_len (int): Maximum length to pad each tensor to.
    
    Returns:
      - padded_embeddings (Tensor): A single tensor with all embeddings padded to max_len.
    """
    padded_embeddings = []
    for emb in embeddings:
        # Pad each tensor to max_len along the sequence dimension (dim=0)
        padding = (0, 0, 0, max_len - emb.shape[0])  # (last dim pad, first dim pad)
        padded_embeddings.append(torch.nn.functional.pad(emb, padding))
    return torch.stack(padded_embeddings)

def extract_and_save_embeddings(data):
    """
    Extracts `embedding_before_classifier` and `embedding_after_classifier`
    from each batch and saves them to separate files in a new directory.
    
    Parameters:
      - data (dict): Loaded data with file paths as keys and lists of batches as values.
      - new_base_dir (str): New base directory to save extracted embeddings.
    """
    new_base_dir = "output/{}/{}/{}/".format(dataset, model, flag)
    for file_path, batches in data.items():
        before_embeddings = []
        after_embeddings = []
        
        # Extract embeddings
        for batch_data in batches:
            if isinstance(batch_data, dict):
                before_embeddings.append(batch_data['embedding_before_classifier'])
                after_embeddings.append(batch_data['embedding_after_classifier'])
        
        # # Convert lists to tensors
        # before_embeddings = torch.stack(before_embeddings)
        # after_embeddings = torch.stack(after_embeddings)
        # Find the maximum sequence length for padding
        max_len_before = max(emb.shape[0] for emb in before_embeddings)
        max_len_after = max(emb.shape[0] for emb in after_embeddings)

        # Pad embeddings to the maximum length
        before_embeddings = pad_embeddings(before_embeddings, max_len_before)
        after_embeddings = pad_embeddings(after_embeddings, max_len_after)
        
        # Determine original folder structure for saving
        relative_path = os.path.relpath(file_path, start=new_base_dir)
        folder_path = os.path.dirname(relative_path)
        
        # Create corresponding folder in new_base_dir
        new_folder_path = os.path.join(new_base_dir, folder_path)
        # import pdb; pdb.set_trace()
        os.makedirs(new_folder_path, exist_ok=True)
        
        # Define new file paths for saving embeddings
        before_file_path = os.path.join(new_folder_path, "embedding_before_classifier.pt")
        after_file_path = os.path.join(new_folder_path, "embedding_after_classifier.pt")
        
        # Save embeddings to new files
        torch.save(before_embeddings, before_file_path)
        torch.save(after_embeddings, after_file_path)
        
        print(f"Saved `embedding_before_classifier` to {before_file_path}")
        print(f"Saved `embedding_after_classifier` to {after_file_path}")

# Example usage
# First, load your data
# flag = 'train'  # or 'test'

# client_id = 1  # specify client ID if needed
# current_round = 10  # specify round if needed
# loaded_data = load_all_saved_data(flag=flag, client_id=client_id, current_round=current_round)

for current_round in range(10,260,10):
    loaded_data = load_all_saved_data(flag=flag, current_round=current_round)
    extract_and_save_embeddings(loaded_data)
# Then, extract and save embeddings to the new directory
# extract_and_save_embeddings(loaded_data)
