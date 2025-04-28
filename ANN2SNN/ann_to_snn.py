from keras.models import Model as KerasModel
from keras.layers import Dense
import numpy as np
import time
import os
import csv

from nxsdk_modules_ncl.dnn.src.dnn_layers import NxInputLayer, NxDense, NxModel
from nxsdk_modules_ncl.dnn.src.utils import to_integer
from nxsdk_modules_ncl.dnn.composable.composable_dnn import ComposableDNN as DNN
from nxsdk_modules_ncl.input_generator.input_generator import InputGenerator
from nxsdk.composable.model import Model as NxSystemModel

def is_feedforward(model):
    return all(isinstance(layer, Dense) for layer in model.layers if hasattr(layer, 'weights'))

def is_ppo_policy(model):
    try:
        from stable_baselines3.ppo.ppo import PPO as PPO_Model
        return isinstance(model, PPO_Model)
    except ImportError:
        return False

# Load PPO model and extract policy
def load_ppo_model(model):    
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
    if not is_feedforward(ann_model):
        raise ValueError("Only fully connected (Dense) layers are supported in the ANN for conversion.")

    input_shape = ann_model.input_shape[1:]

    # Create input layer
    nx_input = NxInputLayer(input_shape=input_shape, vThMant=vth_mant, biasExp=bias_exp)
    x = nx_input.input

    # Add Dense layers to the SNN
    for layer in ann_model.layers:
        if isinstance(layer, Dense):
            output_dim = layer.output_shape[-1]
            x = NxDense(output_dim, vThMant=vth_mant,
                        weightExponent=weight_exponent,
                        synapseEncoding=synapse_encoding)(x)

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


def transfer_ann_weights_to_snn(ann_model, snn_model, weight_bits=8):
    """
    Transfers ANN weights to SNN model after converting to integer format.
    
    Args:
        ann_model (KerasModel): Trained ANN model.
        snn_model (NxModel): Target SNN model.
        weight_bits (int): Bit-width for fixed-point conversion.
    """
    weights = ann_model.get_weights()
    weights_only = weights[0::2]
    biases_only = weights[1::2]

    snn_weights = []
    for w, b in zip(weights_only, biases_only):
        w_int, b_int = to_integer(w, b, weight_bits)
        snn_weights.extend([w_int, b_int])
    
    snn_model.set_weights(snn_weights)
    
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
    
    snn_system.run(num_steps_per_sample * num_samples, aSync=True)

    print(f"\nRunning in {'BATCH' if batch_mode else 'SINGLE'} mode (Batch Size: {batch_size if batch_mode else 1})...")
    tStartInput = time.time()  # Time before input encoding
    
    if batch_mode:
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
        input_encoding_time = tEndInput - tStartInput
        classification_time = tEndClassification - tStartClassification
        total_execution_time = tEndClassification - tStart
        print("\nBenchmarking Results:")
        print(f"Mode: {'Batch' if batch_mode else 'Single'}")
        print(f"Batch Size: {batch_size if batch_mode else 'N/A'}")
        print(f"Num Steps per Sample: {num_steps_per_sample}")
        print(f"Num Samples: {num_samples}")
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
            policy = load_ppo_model(ann_model)
                
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
