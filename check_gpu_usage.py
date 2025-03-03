import torch
from transformers import ViltModel

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the ViLT model
model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

# Move model to GPU
model.to(device)

# Display memory summary
print(torch.cuda.memory_summary(device=device))
