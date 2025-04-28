from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, AveragePooling2D, Flatten
import numpy as np

from stable_baselines3 import PPO

from ann_to_snn import SNNConverter

def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

def initialize_mlp_with_random_weights(input_size, layer_sizes):
    """
    Build a feedforward ANN and initialize with Xavier random weights.
    
    Args:
        input_size (int): Size of input layer.
        layer_sizes (list): List of hidden and output layer sizes.
        
    Returns:
        model (Keras Model): ANN with random weights.
    """
    
    print("Building MLP Model...")
    model = Sequential(name="MLP_Model")
    model.add(Input(shape=(input_size,), name="mlp_input"))
    for i, size in enumerate(layer_sizes):
        model.add(Dense(size, activation='relu', name=f"mlp_dense_{i}"))
    model.build(input_shape=(None, input_size))

    # Xavier initialization (applied after building all layers)
    weights = []
    prev_size = input_size
    for i, size in enumerate(layer_sizes):
        w = xavier_init(prev_size, size)
        b = np.zeros(size)
        weights.extend([w, b])
        prev_size = size
    model.set_weights(weights)

    model.summary()
    return model

# --- CNN Test Setup ---

def build_simple_cnn(input_shape, num_classes):
    """
    Builds a simple Keras CNN model for testing.

    Args:
        input_shape (tuple): Input shape (e.g., H, W, C).
        num_classes (int): Number of output classes.

    Returns:
        model (Keras Model): Simple CNN model.
    """
    print("Building CNN Model...")
    model = Sequential(name="CNN_Model")
    model.add(Input(shape=input_shape, name="cnn_input"))

    # Conv -> Pool Block 1
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same', name="cnn_conv1"))
    model.add(AveragePooling2D(pool_size=(2, 2), name="cnn_pool1"))

    # Conv -> Pool Block 2
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', name="cnn_conv2"))
    model.add(AveragePooling2D(pool_size=(2, 2), name="cnn_pool2"))

    # Flatten and Dense Layers
    model.add(Flatten(name="cnn_flatten"))
    model.add(Dense(32, activation='relu', name="cnn_dense1"))
    model.add(Dense(num_classes, activation='relu', name="cnn_output")) # Use ReLU for output to match SNN firing counts better than softmax

    model.summary()
    return model

# --- Test Execution ---

# 1. MLP Test
print("\n" + "="*30)
print("--- Running MLP Test ---")
print("="*30)

# Create MLP ANN
mlp_input_size = 784
mlp_layer_sizes = [128, 64, 10]
mlp_model = initialize_mlp_with_random_weights(mlp_input_size, mlp_layer_sizes)

# Convert MLP to SNN system
print("\nConverting MLP to SNN...")
mlp_snn = SNNConverter(mlp_model, num_steps_per_sample=512)

# Boot MLP SNN
print("\nBooting MLP SNN...")
mlp_snn.boot()

# Generate dummy inputs for MLP and run inference/comparison
print("\nRunning MLP ANN vs SNN Comparison...")
num_mlp_samples = 50
dummy_mlp_inputs = np.random.uniform(0, 1, (num_mlp_samples, mlp_input_size))
dummy_mlp_inputs_int = (dummy_mlp_inputs * 255).astype(int) # Scale and convert to int

# Compare ANN vs SNN outputs
mlp_snn.compare_to_ann(dummy_mlp_inputs_int, batch_mode=False, print_summary=True) 

# # 2. CNN Test
# print("\n" + "="*30)
# print("--- Running CNN Test ---")
# print("="*30)

# # Define CNN parameters
# cnn_input_shape = (28, 28, 1) # Example: MNIST-like grayscale
# cnn_num_classes = 10

# # Build CNN model (uses default Keras weight initializers)
# cnn_model = build_simple_cnn(cnn_input_shape, cnn_num_classes)

# # Convert CNN to SNN system
# print("\nConverting CNN to SNN...")
# # Use more steps for CNN potentially, adjust as needed for speed/accuracy trade-off
# cnn_snn = SNNConverter(cnn_model, num_steps_per_sample=1024)

# # Boot CNN SNN
# print("\nBooting CNN SNN...")
# cnn_snn.boot()

# # Generate dummy inputs for CNN and run inference/comparison
# print("\nRunning CNN ANN vs SNN Comparison...")
# num_cnn_samples = 20 # Use fewer samples for CNN test as it's slower
# dummy_cnn_inputs = np.random.uniform(0, 1, (num_cnn_samples,) + cnn_input_shape)
# dummy_cnn_inputs_int = (dummy_cnn_inputs * 255).astype(int) # Scale and convert to int

# # Compare ANN vs SNN outputs
# cnn_snn.compare_to_ann(dummy_cnn_inputs_int, batch_mode=False, print_summary=True)


# print("\n--- Test Script Finished ---")

# if __name__ == "__main__":
#     # from gym_multigrid.envs.ctf import CtfMvNEnv

#     # set_random_seed()
    
#     # Change model path based on the policy you want to test
#     model_path = "../RL_Game/gym-multigrid/out/models/vanilla_snnppo_2v2_ctf_{}"
#     # model_path = "out/models/vanilla_snnppo_2v2_ctf_PatrolFightPolicy_{}"
    
#     network_size = 64
    
#     n_runs = 20
#     num_steps_per_sample = 256
#     render = False
    
#     model_path=model_path.format(network_size)
    
#     model = PPO.load(model_path.format(network_size),
#                     custom_objects=dict(policy_class=CustomActorCriticPolicy))
    
#     snn = SNNConverter(ann_model=model,
#                        num_steps_per_sample=num_steps_per_sample)
    
#     snn.boot()
    
    
#     print(snn)