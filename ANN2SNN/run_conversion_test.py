from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np

from stable_baselines3 import PPO

def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

def initialize_ann_with_random_weights(input_size, layer_sizes):
    """
    Build a feedforward ANN and initialize with Xavier random weights.
    
    Args:
        input_size (int): Size of input layer.
        layer_sizes (list): List of hidden and output layer sizes.
        
    Returns:
        model (Keras Model): ANN with random weights.
    """
    model = Sequential()
    model.add(Input(shape=(input_size,)))
    for size in layer_sizes:
        model.add(Dense(size, activation='relu'))

    # Xavier initialization
    weights = []
    prev_size = input_size
    for size in layer_sizes:
        w = xavier_init(prev_size, size)
        b = np.zeros(size)
        weights.extend([w, b])
        prev_size = size
    model.set_weights(weights)
    
    model.summary()
    return model

from ann_to_snn import SNNConverter
import numpy as np

# Create ANN with random weights
input_size = 128
layer_sizes = [64, 32, 10]
ann_model = initialize_ann_with_random_weights(input_size, layer_sizes)

# Convert to SNN system
snn = SNNConverter(ann_model)

# Boot SNN
snn.boot()

# Generate dummy inputs and run inference
dummy_inputs = np.random.uniform(0, 1, (100, input_size))
dummy_inputs = (dummy_inputs * 255)

# snn_outputs = snn.run(dummy_inputs)
snn.compare_to_ann(dummy_inputs, batch_mode=False, print_summary=True)

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
    
    

