import os
import json

def load_config(config_file='neurosync_config.py'):
    # Load the NeuroSync configuration from the specified file
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, config_file='neurosync_config.py'):
    # Save the NeuroSync configuration to the specified file
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def get_machine_learning_model_path(config):
    # Get the path to the machine learning model from the configuration
    return config['machine_learning_model_path']

def get_quantum_key_length(config):
    # Get the length of the quantum key from the configuration
    return config['quantum_key_length']

def get_error_correction_code(config):
    # Get the error correction code from the configuration
    return config['error_correction_code']

def get_database_path(config):
    # Get the path to the database from the configuration
    return config['database_path']

def get_logging_level(config):
    # Get the logging level from the configuration
    return config['logging_level']

def get_logging_path(config):
    # Get the path to the log file from the configuration
    return config['logging_path']

def get_debug_mode(config):
    # Get the debug mode from the configuration
    return config['debug_mode']
