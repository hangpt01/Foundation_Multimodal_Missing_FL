import numpy as np
import random

class MockGenerator:
    def __init__(self, num_clients, num_classes, num_samples):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.train_data = self.generate_mock_data(num_samples, num_classes)
        self.num_clients = num_clients
        self.taskpath = './notebook'
    
    def generate_mock_data(self, num_samples, num_classes):
        # Generate random multi-label data where each sample can have multiple labels
        train_data = []
        for i in range(num_samples):
            # Randomly generate labels (multi-label format, binary array)
            labels = np.random.randint(0, 2, size=(num_classes,))
            train_data.append({'labels': labels})
        return train_data
    
    def get_taskname(self):
        return 'Mock Multi-Label Task'


def generate_iid_partition(generator):
    import numpy as np
    print(generator)
    
    # Ensure that the labels are in numpy array format
    # labels = np.array(generator.train_data.labels)  # Convert labels to numpy array if it's a list
    labels = np.array([sample['labels'] for sample in generator.train_data])  # Convert list of labels to numpy array
    import pdb; pdb.set_trace()
    # Initialize the local data list for each client
    local_datas = [[] for _ in range(generator.num_clients)]
    
    # Iterate over each of the 23 labels (assuming the dataset has up to 23 possible labels)
    for label in range(generator.num_classes):  # assuming num_classes is 23
        # Find the indices of samples that have this label (multi-label case)
        label_indices = np.where(labels[:, label] == 1)[0]  # multi-labels are binary vectors
        
        # Randomly shuffle the indices for this label
        permutation = np.random.permutation(label_indices)
        
        # Split the permuted indices equally across all clients
        split = np.array_split(permutation, generator.num_clients)
        
        # Assign the split indices to the respective clients
        for i, idxs in enumerate(split):
            local_datas[i] += idxs.tolist()  # Add the indices to each client's dataset
            
    # Optional: Shuffle each client's local data to ensure randomness
    for i in range(generator.num_clients):
        local_datas[i] = np.random.permutation(local_datas[i]).tolist()

    return local_datas


def visualize_by_class_imdb(self, train_cidxs):
    import matplotlib.pyplot as plt
    import matplotlib.colors
    import numpy as np
    import random
    import os

    # Create figure for plotting
    fig_width = 15  # Adjust width if necessary
    fig_height = 6 + self.num_clients * 0.5  # Adjust height based on number of clients
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Set colors for the different classes
    colors = [key for key in matplotlib.colors.CSS4_COLORS.keys()]
    random.shuffle(colors)  # Shuffle colors for random class-color assignment

    client_height = 1  # Height of each client bar

    # Number of samples per client
    data_columns = [len(cidx) for cidx in train_cidxs]
    row_map = {k: i for k, i in zip(np.argsort(data_columns), [_ for _ in range(self.num_clients)])}
    # import pdb; pdb.set_trace()
    # print([len(i) for i in train_cidxs])
    # Iterate over each client to visualize the class distribution
    for cid, cidxs in enumerate(train_cidxs):  # list of all clients' sample indices
        # Extract multi-label information for each sample
        labels_list = [self.train_data[did]['labels'] for did in cidxs]  # Assuming 'labels' is a list of 23 elements
        
        # Initialize a counter for each class across the multi-label data
        lb_counter = np.zeros(self.num_classes)  # self.num_classes = 23 in this case

        # Count the occurrence of each label across all samples for the client
        for labels in labels_list:
            labels_array = np.array(labels)  # Ensure labels are in numpy array form
            lb_counter += labels_array  # Accumulate label counts

        offset = 0
        y_bottom = row_map[cid] - client_height / 2.0
        y_top = row_map[cid] + client_height / 2.0

        # Plot bars for each class
        for lbi in range(self.num_classes):  # Iterate over 23 classes (labels)
            if lb_counter[lbi] > 0:  # Only plot if this label has any data
                plt.fill_between([offset, offset + lb_counter[lbi]], y_bottom, y_top, facecolor=colors[lbi], edgecolor='black', label=f'Class {lbi}' if cid == 0 else "")
                offset += lb_counter[lbi]  # Increment offset for next label

        # Add total number of samples for each client as text in the middle of the bar
        total_samples = len(cidxs)  # Total number of samples for this client
        plt.text(offset / 2, (y_bottom + y_top) / 2, f'Total: {total_samples}', ha='center', va='center', color='black', fontsize=10)

    # Set x-axis limits based on the maximum number of samples any client has
    max_samples = max([sum(np.array([self.train_data[did]['labels'] for did in cidxs]).sum(axis=0)) for cidxs in train_cidxs])
    plt.xlim(0, max_samples)
    plt.ylim(-0.5, len(train_cidxs) - 0.5)

    # Set axis labels
    plt.ylabel('Client ID')
    plt.xlabel('Number of Samples per Class')

    # Add title
    plt.title(self.get_taskname())

    # Adjust layout to create more space between the plot and the legend
    plt.subplots_adjust(bottom=0.3, top=0.9)  # Increase bottom space for legend and push plot up

    # Adjust legend - Multiple columns to handle large number of labels, with more space below
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=6, fontsize='small', frameon=False)

    # Save the figure as an image file
    plt.savefig(os.path.join(self.taskpath, 'data_dist_imdb.jpg'))
    plt.show()

# Initialize mock data
num_clients = 5
num_classes = 23
num_samples = 100

# Create mock generator with data
generator = MockGenerator(num_clients=num_clients, num_classes=num_classes, num_samples=num_samples)

# Generate IID partition
train_cidxs = generate_iid_partition(generator)

# Call the visualization function to test
visualize_by_class_imdb(generator, train_cidxs)
