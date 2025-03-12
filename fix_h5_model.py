import h5py
import os
import shutil
import json

def fix_h5_model(input_model_path, output_model_path):
    """
    Fix a TensorFlow model by directly modifying the HDF5 file structure
    to replace problematic layer names containing '/' characters.
    
    Args:
        input_model_path: Path to the original model file
        output_model_path: Path to save the fixed model
    """
    print(f"Processing model file: {input_model_path}")
    
    # First, make a copy of the original file
    shutil.copy2(input_model_path, output_model_path)
    
    # Open the copied file for modification
    with h5py.File(output_model_path, 'r+') as h5file:
        # Find and fix problematic layer names in the model_config
        if 'model_config' in h5file.attrs:
            model_config = json.loads(h5file.attrs['model_config'])
            fixed_config = fix_config(model_config)
            h5file.attrs['model_config'] = json.dumps(fixed_config)
            print("Fixed model_config attribute")
        
        # Find and fix problematic layer names in the layer structure
        problematic_layers = []
        
        def visit_func(name, obj):
            if isinstance(obj, h5py.Group) and 'config' in obj.attrs:
                config = json.loads(obj.attrs['config'])
                if 'name' in config and '/' in config['name']:
                    problematic_layers.append((name, config['name']))
                    # Fix the config
                    config['name'] = config['name'].replace('/', '_')
                    obj.attrs['config'] = json.dumps(config)
                    print(f"Fixed layer config: {name}")
        
        h5file.visititems(visit_func)
        
        if problematic_layers:
            print(f"Fixed {len(problematic_layers)} problematic layer names:")
            for layer_path, layer_name in problematic_layers:
                print(f"  - {layer_path}: {layer_name} -> {layer_name.replace('/', '_')}")
        else:
            print("No problematic layer names found in the HDF5 file structure.")
    
    print(f"Fixed model saved to {output_model_path}")
    return True

def fix_config(config):
    """
    Recursively fix layer names in the model config.
    
    Args:
        config: The model configuration dictionary
    
    Returns:
        The fixed configuration dictionary
    """
    if isinstance(config, dict):
        # Fix layer name if present
        if 'name' in config and isinstance(config['name'], str) and '/' in config['name']:
            old_name = config['name']
            config['name'] = config['name'].replace('/', '_')
            print(f"Fixed config name: {old_name} -> {config['name']}")
        
        # Recursively process all dictionary values
        for key, value in config.items():
            config[key] = fix_config(value)
    
    elif isinstance(config, list):
        # Recursively process all list items
        for i, item in enumerate(config):
            config[i] = fix_config(item)
    
    return config

if __name__ == "__main__":
    input_model_path = "cifar10_cnn_model.h5"
    output_model_path = "cifar10_cnn_model_fixed.h5"
    
    success = fix_h5_model(input_model_path, output_model_path)
    
    if success:
        print("Model fixed successfully!")
    else:
        print("Failed to fix the model.") 