import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

# Define some example tensors
batch_size = 2
sequence_length = 5
hidden_size = 768

# Example tensor for last_hidden_state
last_hidden_state_tensor = torch.randn(batch_size, sequence_length, hidden_size)

# Example tensor for pooler_output
pooler_output_tensor = torch.randn(batch_size, hidden_size)

# Example list of hidden states (optional)
hidden_states_list = [torch.randn(batch_size, sequence_length, hidden_size)] * 12  # 12 layers

# Example list of attentions (optional)
attentions_list = [torch.randn(batch_size, 12, sequence_length, sequence_length)] * 12  # 12 layers

# Create an instance of BaseModelOutputWithPooling
output = BaseModelOutputWithPooling(
    last_hidden_state=last_hidden_state_tensor,
    pooler_output=pooler_output_tensor,
    hidden_states=hidden_states_list,
    attentions=attentions_list
)

# Access attributes
last_hidden_state = output.last_hidden_state
pooler_output = output.pooler_output
hidden_states = output.hidden_states
attentions = output.attentions

# Print shapes of tensors
print("Last hidden state shape:", last_hidden_state.shape)
print("Pooler output shape:", pooler_output.shape)

# Print number of layers in hidden states and attentions
num_layers_hidden_states = len(hidden_states) if hidden_states else 0
num_layers_attentions = len(attentions) if attentions else 0
print("Number of layers in hidden states:", num_layers_hidden_states)
print("Number of layers in attentions:", num_layers_attentions)
