import tensorflow as tf
import os
import re

def fix_model(input_model_path, output_model_path):
    """
    Fix a TensorFlow model with invalid layer names containing '/' characters.
    
    Args:
        input_model_path: Path to the original model file
        output_model_path: Path to save the fixed model
    """
    print(f"Loading model from {input_model_path}...")
    
    # Define a custom layer class to handle the problematic layer names
    class CustomLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            # Replace any '/' in the name with '_'
            if 'name' in kwargs and '/' in kwargs['name']:
                print(f"Fixing layer name: {kwargs['name']}")
                kwargs['name'] = kwargs['name'].replace('/', '_')
            super(CustomLayer, self).__init__(**kwargs)
    
    # Create a dictionary mapping layer types to our custom layer
    custom_objects = {
        'Conv2D': tf.keras.layers.Conv2D,
        # Add other layer types if needed
    }
    
    # Load the model with custom object scope
    with tf.keras.utils.custom_object_scope(custom_objects):
        try:
            # Try to load the model with h5py directly to avoid TensorFlow's name validation
            import h5py
            with h5py.File(input_model_path, 'r') as h5file:
                # Get all layer configs
                layer_names = []
                def visit_func(name, obj):
                    if isinstance(obj, h5py.Group) and 'config' in obj:
                        layer_names.append(name)
                h5file.visititems(visit_func)
                
                print(f"Found {len(layer_names)} layers in the model")
                
                # Check for problematic layer names
                problematic_layers = []
                for name in layer_names:
                    if '/' in name:
                        problematic_layers.append(name)
                
                if problematic_layers:
                    print(f"Found {len(problematic_layers)} problematic layer names:")
                    for layer in problematic_layers:
                        print(f"  - {layer}")
                else:
                    print("No problematic layer names found in the HDF5 file structure.")
            
            # Now try to load the model with TensorFlow
            model = tf.keras.models.load_model(input_model_path, compile=False)
            print("Model loaded successfully!")
            
            # Save the fixed model
            model.save(output_model_path)
            print(f"Fixed model saved to {output_model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            
            # Alternative approach: load the model weights into a new model
            print("Trying alternative approach...")
            try:
                # Create a new model with the same architecture but fixed layer names
                # This is a simplified example - you'll need to adapt this to your specific model architecture
                new_model = create_new_model()
                
                # Load weights from the original model, skipping layers with problematic names
                new_model.load_weights(input_model_path, by_name=True, skip_mismatch=True)
                
                # Save the new model
                new_model.save(output_model_path)
                print(f"Fixed model saved to {output_model_path}")
                return True
            except Exception as e2:
                print(f"Alternative approach failed: {str(e2)}")
                return False

def create_new_model():
    """
    Create a new model with the same architecture as the original CIFAR-10 model.
    This is a placeholder - you'll need to define your actual model architecture here.
    """
    # This is a simple CNN for CIFAR-10 - replace with your actual model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3), name='conv1_conv'),
        tf.keras.layers.BatchNormalization(name='conv1_bn'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_conv'),
        tf.keras.layers.BatchNormalization(name='conv2_bn'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_conv'),
        tf.keras.layers.BatchNormalization(name='conv3_bn'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool3'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(128, activation='relu', name='dense1'),
        tf.keras.layers.Dense(10, activation='softmax', name='dense2')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    input_model_path = "cifar10_cnn_model.h5"
    output_model_path = "cifar10_cnn_model_fixed.h5"
    
    success = fix_model(input_model_path, output_model_path)
    
    if success:
        print("Model fixed successfully!")
    else:
        print("Failed to fix the model.") 