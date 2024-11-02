import torch
import os

# or imdb
dataset = "food101"
# or missing_aware
model = "L2P_Prob"
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

def extract_and_save_summarizing_prompts(flag, client_id=None, current_round=None, save_dir="output/summarizing_prompts"):
    """
    Loads data files, extracts `summarizing_prompts` from each batch, and saves them in separate files.
    
    Parameters:
      - flag (str): Indicates whether to load 'train' or 'test' data.
      - client_id (int, optional): If provided, loads data for a specific client.
      - current_round (int, optional): If provided, loads data for a specific round.
      - save_dir (str): The directory to save extracted summarizing prompts.
    """
    base_path = f"output/{dataset}/{model}"
    os.makedirs(save_dir, exist_ok=True)
    
    if flag == "train":
        train_path = os.path.join(base_path, "train")
        clients = [f"client_{client_id}"] if client_id else os.listdir(train_path)
        
        for client_folder in clients:
            client_path = os.path.join(train_path, client_folder)
            if not os.path.isdir(client_path):
                continue
            
            rounds = [f"sample_data_round_{current_round}.pt"] if current_round else os.listdir(client_path)
            for file_name in rounds:
                file_path = os.path.join(client_path, file_name)
                if os.path.exists(file_path):
                    data = torch.load(file_path)
                    summarizing_prompts_list = []
                    
                    for batch_data in data:
                        if "summarizing_prompts" in batch_data:
                            summarizing_prompts_list.append(batch_data["summarizing_prompts"])
                    
                    if summarizing_prompts_list:
                        # Concatenate all summarizing prompts across batches
                        summarizing_prompts = torch.cat(summarizing_prompts_list, dim=0)
                        
                        # Define save path for summarizing prompts
                        relative_path = os.path.relpath(file_path, base_path)
                        save_path = os.path.join(save_dir, relative_path)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        
                        # Save summarizing prompts to new file
                        torch.save(summarizing_prompts, save_path)
                        print(f"Saved summarizing prompts to {save_path} with shape {summarizing_prompts.shape}")
    
    elif flag == "test":
        test_path = os.path.join(base_path, "test")
        rounds = [f"sample_data_round_{current_round}.pt"] if current_round else os.listdir(test_path)
        
        for file_name in rounds:
            file_path = os.path.join(test_path, file_name)
            if os.path.exists(file_path):
                data = torch.load(file_path)
                summarizing_prompts_list = []
                
                for batch_data in data:
                    if "summarizing_prompts" in batch_data:
                        summarizing_prompts_list.append(batch_data["summarizing_prompts"])
                
                if summarizing_prompts_list:
                    # Concatenate all summarizing prompts across batches
                    summarizing_prompts = torch.cat(summarizing_prompts_list, dim=0)
                    
                    # Define save path for summarizing prompts
                    relative_path = os.path.relpath(file_path, base_path)
                    save_path = os.path.join(save_dir, relative_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    # Save summarizing prompts to new file
                    torch.save(summarizing_prompts, save_path)
                    print(f"Saved summarizing prompts to {save_path} with shape {summarizing_prompts.shape}")
    
    else:
        print(f"Invalid flag: {flag}. Use 'train' or 'test'.")
# Example usage
# First, load your data
# flag = 'train'  # or 'test'

# client_id = 1  # specify client ID if needed
# current_round = 10  # specify round if needed
# loaded_data = load_all_saved_data(flag=flag, client_id=client_id, current_round=current_round)
for current_round in range(10,260,10):
    extract_and_save_summarizing_prompts(flag, current_round=current_round)

# for current_round in range(10,260,10):
#     loaded_data = load_all_saved_data(flag=flag, current_round=current_round)
#     extract_and_save_embeddings(loaded_data)
# Then, extract and save embeddings to the new directory
# extract_and_save_embeddings(loaded_data)
