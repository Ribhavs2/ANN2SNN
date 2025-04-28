from keras.models import Model as KerasModel
from keras.layers import Dense
import numpy as np
import time
import os
import csv

from nxsdk_modules_ncl.dnn.src.dnn_layers import (
    NxInputLayer, NxDense, NxConv2D, NxDepthwiseConv2D,
    NxAveragePooling2D, NxFlatten, NxZeroPadding2D, NxReshape, NxModel
)
from nxsdk_modules_ncl.dnn.src.utils import to_integer
from nxsdk_modules_ncl.dnn.composable.composable_dnn import ComposableDNN as DNN
from nxsdk_modules_ncl.input_generator.input_generator import InputGenerator
from nxsdk.composable.model import Model as NxSystemModel

from keras.layers import Dense, Conv2D, DepthwiseConv2D, AveragePooling2D, Flatten, ZeroPadding2D, Reshape, InputLayer

def is_feedforward(model):
    return all(isinstance(layer, Dense) for layer in model.layers if hasattr(layer, 'weights'))

def is_ppo_policy(model):
    try:
        from stable_baselines3.ppo.ppo import PPO as PPO_Model
        return isinstance(model, PPO_Model)
    except ImportError:
        return False

# Load PPO model and extract policy
def load_ppo_policy(model):    
    policy = model.policy
                
    return policy
    
def get_combined_linear_sizes(policy):
    import torch.nn as nn
    layers = []

    # Handle policy_net (always Sequential)
    for layer in policy.mlp_extractor.policy_net:
        if isinstance(layer, nn.Linear):
            layers.append((layer.in_features, layer.out_features))
    
    # Handle action_net (can be Sequential or single Linear layer)
    action_net = policy.action_net
    if isinstance(action_net, nn.Sequential):
        for layer in action_net:
            if isinstance(layer, nn.Linear):
                layers.append((layer.in_features, layer.out_features))
    elif isinstance(action_net, nn.Linear):
        layers.append((action_net.in_features, action_net.out_features))
    else:
        raise TypeError(f"Unexpected type for action_net: {type(action_net)}")

    return layers


def build_snn_from_ann(ann_model, vth_mant=2**9, bias_exp=6, weight_exponent=0, synapse_encoding='sparse'):
    """
    Builds a spiking neural network (SNN) model from a given ANN model.
    
    Args:
        ann_model (KerasModel): A Keras ANN model (feedforward, using Dense layers).
        vth_mant (int): Voltage threshold mantissa for Nx layers.
        bias_exp (int): Bias exponent for NxInputLayer.
        weight_exponent (int): Weight exponent for NxDense layers.
        synapse_encoding (str): Synapse encoding method (e.g., 'sparse').

    Returns:
        snn_model (NxModel): The constructed SNN model.
    """
    # if not is_feedforward(ann_model):
    #     raise ValueError("Only fully connected (Dense) layers are supported in the ANN for conversion.")

   # Get input shape - handle potential list wrapping if model loaded weirdly
    if isinstance(ann_model.input_shape, list):
         input_shape = ann_model.input_shape[0][1:] # Get shape tuple, drop batch dim
    else:
         input_shape = ann_model.input_shape[1:] # Drop batch dim

    if not input_shape:
        raise ValueError("Could not determine input shape from the ANN model.")
    
    print(f"  Input shape (detected): {input_shape}")

    # Create input layer
    nx_input = NxInputLayer(input_shape=input_shape, vThMant=vth_mant, biasExp=bias_exp)
    x = nx_input.input
    print(f"  Created NxInputLayer with shape: {input_shape}")

    # Add Dense layers to the SNN
    for layer in ann_model.layers:
        layer_type = type(layer).__name__
        print(f"  Processing Keras layer: {layer.name} ({layer_type})")
        
        # Skip the Keras InputLayer itself, already handled by NxInputLayer
        if isinstance(layer, InputLayer):
            print("    Skipping InputLayer (handled by NxInputLayer).")
            continue
        
        # Get common Nx Kwargs
        nx_kwargs = {
            'vThMant': vth_mant,
            'weightExponent': weight_exponent,
            'synapseEncoding': synapse_encoding
            # Add other relevant shared NxKwargs if needed (e.g., numWeightBits handled in transfer)
        }
        config = layer.get_config() # Get Keras layer config
        
        if isinstance(layer, Dense):
            units = config['units']
            # activation = config['activation']
            print(f"    Adding NxDense with units: {units}")
            x = NxDense(units, **nx_kwargs)(x)
        elif isinstance(layer, Conv2D):
            filters = config['filters']
            kernel_size = config['kernel_size']
            strides = config['strides']
            padding = config['padding']
            # activation = config['activation']
            print(f"    Adding NxConv2D with filters: {filters}, kernel: {kernel_size}, strides: {strides}, padding: {padding}")
            x = NxConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, **nx_kwargs)(x)
        elif isinstance(layer, DepthwiseConv2D):
            kernel_size = config['kernel_size']
            strides = config['strides']
            padding = config['padding']
            # depth_multiplier = config['depth_multiplier'] # NxDepthwiseConv2D infers this? Check dnn_layers.py
            # activation = config['activation']
            print(f"    Adding NxDepthwiseConv2D with kernel: {kernel_size}, strides: {strides}, padding: {padding}")
            x = NxDepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding, **nx_kwargs)(x)
        elif isinstance(layer, AveragePooling2D):
            pool_size = config['pool_size']
            strides = config['strides']
            padding = config['padding']
            print(f"    Adding NxAveragePooling2D with pool: {pool_size}, strides: {strides}, padding: {padding}")
            # Note: NxAveragePooling2D doesn't take many nx_kwargs directly in __init__ from dnn_layers.py
            # They might be handled internally or during compilation. Pass basic Keras params.
            x = NxAveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)(x)
            # --> Check NxAveragePooling2D constructor in dnn_layers.py if parameters need adjustment
        elif isinstance(layer, Flatten):
            data_format = config.get('data_format', None) # Use .get for safety
            print(f"    Adding NxFlatten (data_format: {data_format})")
            x = NxFlatten(data_format=data_format)(x)
        elif isinstance(layer, ZeroPadding2D):
            padding = config['padding']
            data_format = config.get('data_format', None)
            print(f"    Adding NxZeroPadding2D with padding: {padding}")
            x = NxZeroPadding2D(padding=padding, data_format=data_format)(x)
        elif isinstance(layer, Reshape):
            target_shape = config['target_shape']
            print(f"    Adding NxReshape with target_shape: {target_shape}")
            x = NxReshape(target_shape=target_shape)(x)
        else:
            print(f"    WARNING: Skipping unsupported Keras layer type: {layer_type}")


    snn_model = NxModel(nx_input.input, x, numCandidatesToCompute=1)
    print("\nSNN Model Summary:")
    print(snn_model.summary())
    return snn_model

def build_snn_from_ann_PPO(input_size, hidden_layer_sizes, output_size, vth_mant=2**9, bias_exp=6, weight_exponent=0):
    """
    Initializes an SNN model with arbitrary hidden layer sizes.
    
    Args:
        input_size (int): Input feature dimension.
        hidden_layer_sizes (List[int]): List of hidden layer output sizes.
        output_size (int): Final output layer size.
        vth_mant (int): Voltage threshold mantissa.
        bias_exp (int): Bias exponent for the input layer.
        weight_exponent (int): Weight exponent for NxDense layers.

    Returns:
        snn_model (NxModel): Constructed SNN model.
    """
    snn_input = NxInputLayer(input_shape=(input_size,), vThMant=vth_mant, biasExp=bias_exp)
    x = snn_input.input
    
    # Create multiple hidden NxDense layers
    for i, hidden_size in enumerate(hidden_layer_sizes):
        x = NxDense(hidden_size, vThMant=vth_mant, weightExponent=weight_exponent)(x)

    # Final output layer
    x = NxDense(output_size, vThMant=vth_mant, weightExponent=weight_exponent)(x)
    
    snn_model = NxModel(snn_input.input, x, numCandidatesToCompute=1)
    print("\nSNN Model Summary:")
    print(snn_model.summary())
    return snn_model

def find_next_keras_trainable_layer(ann_model, start_idx):
    """Helper to find the next Keras layer with trainable weights."""
    for i in range(start_idx, len(ann_model.layers)):
        layer = ann_model.layers[i]
        if hasattr(layer, 'weights') and len(layer.get_weights()) > 0 and layer.trainable:
             # Also check if it's a type we expect to transfer (Dense, Conv)
             if isinstance(layer, (Dense, Conv2D, DepthwiseConv2D)):
                 return i, layer
    return -1, None # Not found

def transfer_ann_weights_to_snn(ann_model, snn_model, weight_bits=8):
    """
    Transfers ANN weights (Dense, Conv2D, DepthwiseConv2D) to the corresponding
    layers in the SNN model, preserving the SNN's non-trainable weights
    (e.g., for AveragePooling). Converts weights to integer format.

    Args:
        ann_model (KerasModel): Trained ANN model (source of trainable weights).
        snn_model (NxModel): Target SNN model (destination).
        weight_bits (int): Bit-width for fixed-point conversion.
    """
    print("Transferring weights from Keras model to SNN...")

    snn_weights_list = snn_model.get_weights()
    final_snn_weights_int = [w.copy() for w in snn_weights_list]

    snn_weight_list_idx = 0  # Tracks position in the flat snn_weights_list
    keras_layer_search_idx = 0 # Where to start searching in Keras layers
    trainable_layers_processed = 0

    # Iterate through SNN layers
    for snn_layer in snn_model.layers:
        snn_layer_type = type(snn_layer).__name__
        snn_layer_has_params = hasattr(snn_layer, 'weights') and len(snn_layer.weights) > 0

        if not snn_layer_has_params:
            print(f"  Skipping SNN layer {snn_layer.name} ({snn_layer_type}) as it has no weights.")
            continue # No weights in SNN layer, move to next SNN layer

        num_weights_in_snn_layer = len(snn_layer.weights)

        # Determine if this SNN layer should receive weights from Keras
        is_trainable_snn_type = isinstance(snn_layer, (NxDense, NxConv2D, NxDepthwiseConv2D))

        if is_trainable_snn_type:
            # Find the next corresponding trainable Keras layer
            keras_layer_idx, keras_layer = find_next_keras_trainable_layer(ann_model, keras_layer_search_idx)

            if keras_layer is not None:
                keras_layer_type = type(keras_layer).__name__
                print(f"  Attempting to match SNN {snn_layer.name} ({snn_layer_type}) with Keras {keras_layer.name} ({keras_layer_type})")

                keras_weights = keras_layer.get_weights()

                # Basic sanity check: Does number of weight arrays match? (Usually 2)
                if len(keras_weights) == num_weights_in_snn_layer:
                    w_keras, b_keras = keras_weights
                    print(f"    Extracting Keras weights - W:{w_keras.shape}, b:{b_keras.shape}")
                    w_int, b_int = to_integer(w_keras, b_keras, weight_bits)

                    # --- Shape Compatibility Check ---
                    expected_w_shape = snn_weights_list[snn_weight_list_idx].shape
                    expected_b_shape = snn_weights_list[snn_weight_list_idx + 1].shape

                    if w_int.shape == expected_w_shape and b_int.shape == expected_b_shape:
                        final_snn_weights_int[snn_weight_list_idx] = w_int
                        final_snn_weights_int[snn_weight_list_idx + 1] = b_int
                        trainable_layers_processed += 1
                        print(f"    Successfully prepared weights for {snn_layer.name}.")
                    else:
                        print(f"    ERROR: Shape mismatch! SNN expects W:{expected_w_shape}, b:{expected_b_shape}. Keras provides W:{w_int.shape}, b:{b_int.shape}. Skipping transfer for this layer.")

                    # Update search index for the *next* Keras layer
                    keras_layer_search_idx = keras_layer_idx + 1

                else:
                    print(f"    WARNING: Weight array count mismatch for Keras layer {keras_layer.name}. Skipping.")

            else:
                print(f"  WARNING: Could not find a matching trainable Keras layer for SNN layer {snn_layer.name}. Using SNN default weights.")

        else: # Handle non-trainable SNN layers (like pooling) that still have weights
            if isinstance(snn_layer, NxAveragePooling2D):
                print(f"  Keeping default non-trainable weights for SNN Pooling layer {snn_layer.name}.")
                # Convert the default SNN weights to integer format
                w_default = snn_weights_list[snn_weight_list_idx]
                b_default = snn_weights_list[snn_weight_list_idx + 1]
                w_default_int, b_default_int = to_integer(w_default, b_default, weight_bits)
                final_snn_weights_int[snn_weight_list_idx] = w_default_int
                final_snn_weights_int[snn_weight_list_idx + 1] = b_default_int
            else:
                 print(f"  WARNING: SNN layer {snn_layer.name} has weights but is not Dense/Conv/Pooling type. Using SNN defaults.")
                 # Convert defaults to int if possible (assuming pairs)
                 if num_weights_in_snn_layer == 2:
                      w_default = snn_weights_list[snn_weight_list_idx]
                      b_default = snn_weights_list[snn_weight_list_idx + 1]
                      w_default_int, b_default_int = to_integer(w_default, b_default, weight_bits)
                      final_snn_weights_int[snn_weight_list_idx] = w_default_int
                      final_snn_weights_int[snn_weight_list_idx + 1] = b_default_int


        # Advance the index into the SNN's flat weight list
        snn_weight_list_idx += num_weights_in_snn_layer


    print(f"Processed {trainable_layers_processed} trainable layers for weight transfer.")

    # Final check on list length
    if len(final_snn_weights_int) != len(snn_weights_list):
         print(f"FATAL ERROR: Final weight list length ({len(final_snn_weights_int)}) does not match expected SNN weights length ({len(snn_weights_list)}).")
         raise ValueError("Weight list construction failed during transfer.")

    # --- Set weights ---
    try:
        snn_model.set_weights(final_snn_weights_int)
        print("Successfully set weights on SNN model.")
    except ValueError as e:
        print(f"\n--- ERROR during snn_model.set_weights ---")
        print(f"  Error message: {e}")
        print(f"  This usually indicates a shape mismatch between a weight array in")
        print(f"  'final_snn_weights_int' and the corresponding weight array expected")
        print(f"  by the SNN layer structure.")
        print(f"  SNN model expected {len(snn_weights_list)} weight arrays.")
        print(f"  Constructed list has {len(final_snn_weights_int)} weight arrays.")

        # Add more detailed debug info: Find the mismatching weight
        expected_weights = snn_model.get_weights() # Get expected shapes again
        for i, (final_w, expected_w) in enumerate(zip(final_snn_weights_int, expected_weights)):
            if final_w.shape != expected_w.shape:
                print(f"\n  Mismatch found at weight index {i}:")
                # Try to find which SNN layer this weight belongs to
                current_idx = 0
                owner_layer = "Unknown"
                for layer in snn_model.layers:
                     if hasattr(layer, 'weights'):
                          num_w = len(layer.weights)
                          if current_idx <= i < current_idx + num_w:
                               owner_layer = f"{layer.name} ({type(layer).__name__})"
                               break
                          current_idx += num_w

                print(f"    Layer: {owner_layer}")
                print(f"    Expected Shape: {expected_w.shape}")
                print(f"    Provided Shape: {final_w.shape}")
                # Try to identify the source Keras layer based on our logic above (difficult after the fact)
                break # Stop after first mismatch
        raise ValueError(f"Weight shape mismatch during set_weights. See details above. Error: {e}")

    except Exception as e:
        print(f"An unexpected error occurred during set_weights: {e}")
        raise e

    
def transfer_ann_weights_to_snn_PPO(policy, snn_model, weight_bits=8):
    """
    Transfers weights from a PPO actor policy to an SNN model.

    Args:
        policy: PPO policy object (with .mlp_extractor.policy_net and .action_net).
        snn_model (NxModel): The target SNN model.
        weight_bits (int): Bit-width for fixed-point conversion.
    """
    import torch.nn as nn

    parameters_int = []

    # Helper: extract and transpose weights from Linear layers
    def extract_and_convert_weights(net):
        for layer in net:
            if isinstance(layer, nn.Linear):
                w = layer.weight.data.cpu().numpy().T  # Transpose to NxSDK format
                b = layer.bias.data.cpu().numpy()
                w_int, b_int = to_integer(w, b, weight_bits)
                parameters_int.extend([w_int, b_int])

    extract_and_convert_weights(policy.mlp_extractor.policy_net)

    action_net = policy.action_net
    if isinstance(action_net, nn.Sequential):
        extract_and_convert_weights(action_net)
    elif isinstance(action_net, nn.Linear):
        w = action_net.weight.data.cpu().numpy().T
        b = action_net.bias.data.cpu().numpy()
        w_int, b_int = to_integer(w, b, weight_bits)
        parameters_int.extend([w_int, b_int])
    else:
        raise TypeError("Unsupported action_net type")

    print(f"Transferred {len(parameters_int)//2} layers to SNN (weights + biases).")
    snn_model.set_weights(parameters_int)
    
def create_snn_system(snn_model, input_size, num_steps_per_sample, input_generator_interval, bias_exp=6):
    """
    Create the full SNN system with input generator and composable DNN.
    
    Args:
        snn_model (NxModel): The SNN model created from the ANN.
        input_size (int): Number of input neurons.
        num_steps_per_sample (int): Number of time steps per input sample.
        input_generator_interval (int): Time interval between input spikes.
        bias_exp (int): Bias exponent used in the input generator.
        
    Returns:
        Tuple[NxSystemModel, DNN, InputGenerator]: 
            - The complete SNN system.
            - The ComposableDNN object.
            - The InputGenerator object.
    """
    dnn = DNN(model=snn_model, num_steps_per_img=num_steps_per_sample)
    input_generator = InputGenerator(shape=(input_size,), interval=input_generator_interval)
    input_generator.setBiasExp(bias_exp)
    
    snn_system = NxSystemModel("snn_inference")
    snn_system.add(dnn)
    snn_system.add(input_generator)
    input_generator.connect(dnn)
    
    return snn_system, dnn, input_generator

def convert_ann_to_snn_system(ann_model, 
                              num_steps_per_sample=1024, 
                              input_generator_interval=1024,
                              vth_mant=2**9, 
                              bias_exp=6, 
                              weight_exponent=0,
                              synapse_encoding='sparse',
                              weight_bits=8):
    """
    Converts a trained ANN to an SNN system ready for simulation.
    
    Args:
        ann_model (Keras Model): Trained feedforward ANN model.
        num_steps_per_sample (int): Time steps for each input sample.
        input_generator_interval (int): Spike interval for the input generator.
        vth_mant (int): Threshold mantissa for spiking neurons.
        bias_exp (int): Bias exponent for input layer and input generator.
        weight_exponent (int): Weight exponent for SNN layers.
        synapse_encoding (str): Encoding strategy ('sparse' or 'dense').
        weight_bits (int): Bit precision for quantizing ANN weights.

    Returns:
        Tuple[NxSystemModel, DNN, InputGenerator]: 
            - The complete SNN system (NxSystemModel).
            - The composable DNN object.
            - The input generator object.
    """
    input_size = ann_model.input_shape[1]
    
    # Step 1: Build SNN
    snn_model = build_snn_from_ann(ann_model,
                                   vth_mant=vth_mant,
                                   bias_exp=bias_exp,
                                   weight_exponent=weight_exponent,
                                   synapse_encoding=synapse_encoding)
    
    # Step 2: Transfer weights
    transfer_ann_weights_to_snn(ann_model, snn_model, weight_bits=weight_bits)
    
    # Step 3: Wrap SNN with input generator system
    snn_system, dnn, input_gen = create_snn_system(snn_model,
                                                   input_size=input_size,
                                                   num_steps_per_sample=num_steps_per_sample,
                                                   input_generator_interval=input_generator_interval,
                                                   bias_exp=bias_exp)
    
    return snn_system, dnn, input_gen

def run_snn(snn_system, dnn, input_generator, dummy_inputs, num_steps_per_sample, num_samples, 
            batch_mode=False, batch_size=None, log=False, log_file="snn_benchmark.csv", print_summary=True):
    """Run the SNN model with support for mini-batch inference, benchmark execution time, and log results."""
    
    tStart = time.time()  # Start time logging
    # snn_system.compile()
    # snn_system.start(snn_system.board)
    # tEndBoot = time.time()  # Time after boot
    
    snn_system.run(num_steps_per_sample * num_samples, aSync=True)

    print(f"\nRunning in {'BATCH' if batch_mode else 'SINGLE'} mode (Batch Size: {batch_size if batch_mode else 1})...")
    tStartInput = time.time()  # Time before input encoding
    
    if batch_mode:
        # scaled_inputs = (dummy_inputs * 255).astype(int)
        int_inputs = dummy_inputs.astype(int)
        
        num_batches = int(np.ceil(num_samples / batch_size))  # Number of batches needed
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = int_inputs[start_idx:end_idx]
            
            print(f"Processing Batch {i+1}/{num_batches} ({len(batch_data)} samples)...")
            input_generator.batchEncode(batch_data)  # Process one batch at a time

    else:
        for i, sample in enumerate(dummy_inputs):
            # scaled_sample = (sample * 255).astype(int)
            int_inputs = sample.astype(int)
            input_generator.encode(int_inputs)  # Process one input at a time
            
    tEndInput = time.time()  # Time after input encoding
    
    print("Waiting for classification to finish...")
    tStartClassification = time.time()  # Time before readout
    snn_outputs = list(dnn.readout_channel.read(num_samples))
    tEndClassification = time.time()  # Time after readout
    snn_system.finishRun()
    snn_system.board.disconnect()
    


    # Print benchmark results
    if print_summary:
        # Compute execution times
        # boot_time = tEndBoot - tStart
        input_encoding_time = tEndInput - tStartInput
        classification_time = tEndClassification - tStartClassification
        total_execution_time = tEndClassification - tStart
        print("\nBenchmarking Results:")
        print(f"Mode: {'Batch' if batch_mode else 'Single'}")
        print(f"Batch Size: {batch_size if batch_mode else 'N/A'}")
        print(f"Num Steps per Sample: {num_steps_per_sample}")
        print(f"Num Samples: {num_samples}")
        # print(f"Boot Time: {boot_time:.4f} seconds")
        print(f"Input Encoding Time: {input_encoding_time:.4f} seconds")
        print(f"Classification Time: {classification_time:.4f} seconds")
        print(f"Total Execution Time: {total_execution_time:.4f} seconds")

    if log:
        # Log results to CSV file
        log_exists = os.path.exists(log_file)
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not log_exists:
                writer.writerow(["Batch Mode", "Batch Size", "Num Steps per Sample", "Num Samples", 
                                 "Input Encoding Time (s)", 
                                 "Classification Time (s)", "Total Execution Time (s)"])
            writer.writerow(["Batch" if batch_mode else "Single", batch_size if batch_mode else "N/A", 
                             num_steps_per_sample, num_samples, 
                             input_encoding_time, classification_time, total_execution_time])

    return np.array(snn_outputs)


class SNNConverter:
    """
    Converts a trained Keras ANN model into a full Loihi-compatible SNN system.
    Encapsulates the system, DNN, and input generator into a single object.
    """

    def __init__(self, ann_model, 
                 num_steps_per_sample=1024,
                #  input_generator_interval=1024,
                 vth_mant=2**9,
                 bias_exp=6,
                 weight_exponent=0,
                 synapse_encoding='sparse',
                 weight_bits=8):
        """
        Initializes the SNNConverter and builds the system.
        """
        input_generator_interval=num_steps_per_sample
        
        self.ann_model = ann_model
        self.num_steps_per_sample = num_steps_per_sample
        
        if is_ppo_policy(ann_model):
            self.is_PPO=True
            policy = load_ppo_policy(ann_model)
                
            layer_sizes = get_combined_linear_sizes(policy)
            self.input_size = layer_sizes[0][0]
            hidden_sizes = [out for (_, out) in layer_sizes[:-1]]
            output_size = layer_sizes[-1][1]

            self.snn_model = build_snn_from_ann_PPO(self.input_size, hidden_sizes, output_size,
                                            vth_mant=vth_mant, bias_exp=bias_exp, weight_exponent=weight_exponent)
            
            transfer_ann_weights_to_snn_PPO(policy, self.snn_model, weight_bits)
        
        else:
            self.is_PPO=False
            self.input_size = ann_model.input_shape[1]

            # Build SNN
            self.snn_model = build_snn_from_ann(ann_model, vth_mant, bias_exp, weight_exponent, synapse_encoding)
        
            # Transfer Weights
            transfer_ann_weights_to_snn(ann_model, self.snn_model, weight_bits)

        # Create System
        self.snn_system, self.dnn, self.input_generator = create_snn_system(
            self.snn_model,
            input_size=self.input_size,
            num_steps_per_sample=num_steps_per_sample,
            input_generator_interval=input_generator_interval,
            bias_exp=bias_exp
        )
    
    def boot(self):
        """Boots the SNN system."""
        self.snn_system.compile()
        self.snn_system.start(self.snn_system.board)
        print("SNN system booted.")

    def run(self, dummy_inputs, batch_mode=False, batch_size=None, log=False, log_file="snn_benchmark.csv", print_summary=True):
        """Runs inference on the SNN system and returns the outputs."""
        return run_snn(
            self.snn_system,
            self.dnn,
            self.input_generator,
            dummy_inputs,
            self.num_steps_per_sample,
            num_samples=len(dummy_inputs),
            batch_mode=batch_mode,
            batch_size=batch_size,
            log=log,
            log_file=log_file,
            print_summary=print_summary
        )
    
    def compare_to_ann(self, inputs, batch_mode=False, batch_size=None,
                       log=False, log_file="snn_benchmark.csv", print_summary=False):
        """
        Compares ANN outputs vs. SNN outputs using argmax and prints accuracy.

        Args:
            inputs (np.ndarray): Input samples (num_samples x input_dim).
            scale_inputs (bool): Whether to scale inputs to [0, 255] for the ANN.
            batch_mode (bool): Whether to run SNN in batch mode.
            batch_size (int): Batch size for SNN (only needed if batch_mode is True).
            log (bool): Whether to log performance results.
            log_file (str): Log file name.
            print_summary (bool): Whether to print benchmark summary.
        """
        ann_outputs = self.ann_model.predict(inputs)
        ann_predictions = np.argmax(ann_outputs, axis=1)

        snn_outputs = self.run(inputs, batch_mode=batch_mode, batch_size=batch_size, log=log, log_file=log_file, print_summary=print_summary)
        snn_predictions = np.array(snn_outputs)

        accuracy = np.mean(ann_predictions == snn_predictions)

        print("ANN vs SNN Comparison:")
        print("ANN Predictions:", ann_predictions)
        print("SNN Outputs    :", snn_predictions)
        print(f"Accuracy      : {accuracy * 100:.2f}%")

        return accuracy
