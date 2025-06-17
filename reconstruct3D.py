
import tensorflow as tf
import keras.backend as K

your_segmentation_model = None # Global variable to hold the loaded model

# --- 1. Custom Metrics/Loss ---
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def iou(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# --- 2. Mount Google Drive ---
from google.colab import drive
# drive.mount('/content/drive')
drive.mount('/content/drive', force_remount=True)

# --- 3. Load the Model ---
MODEL_PATH = '/content/drive/MyDrive/colab_checkpoints/model_checkpoint.h5'
# Optional: Specify a Google Drive path for your MRI slices.
# Example: MRI_DATA_SOURCE_PATH = '/content/drive/MyDrive/my_mri_data_folder'
# If set to None or an empty string, the script will prompt for a local upload.
MRI_DATA_SOURCE_PATH = '/content/drive/MyDrive/colab_checkpoints/dataset'

try:
    print(f"Trying to load model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'dice_coef': dice_coef, 'iou': iou, 'dice_coef_loss': dice_coef_loss}
    )
    print("✅ Model loaded successfully!")
    your_segmentation_model = model
except Exception as e:
    print(f"⚠️ Could not load model: {e}")



# --- Cell 1: Remaining Required Libraries ---
# These are usually pre-installed in Google Colab, but kept for clarity.
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from google.colab import files # Required for file uploads in Colab
import zipfile # To handle unzipping uploaded directories

# --- Configuration ---
# TUMOR_THRESHOLD:
# This threshold is now applied to the *output of your model*.
# If your model outputs *probabilistic maps* (e.g., values from 0.0 to 1.0),
# this threshold binarizes that probability. A common value is 0.5.
# If your model outputs *perfect binary masks* (e.g., 0 for background, 255 for tumor, or just 0/1),
# set this value to a very low number like 0.1 or 1 to ensure all tumor pixels are captured.
TUMOR_THRESHOLD = 100 # <<<--- ADJUST THIS VALUE BASED ON YOUR MODEL'S OUTPUT CHARACTERISTICS

# Output filename for the .obj mesh. This file will be created in the Colab environment.
OUTPUT_OBJ_FILENAME = "tumor_reconstruction.obj" # <<<--- You can change this name

# --- Cell 2: Upload your .tif MRI slices (Run this after Cell 1) ---
# IMPORTANT:
# 1. Compress your directory of .tif MRI slices into a .zip file on your local machine FIRST.
#    E.g., if your slices are in a folder named 'my_mri_slices', zip 'my_mri_slices' into 'my_mri_slices.zip'.
# 2. Upload the .zip file using the button that appears after running this cell.
# 3. Specify the name of the unzipped directory for IMAGE_DIRECTORY.

def upload_and_unzip_mri_data():
    """
    Prompts the user to upload a zip file, unzips it, and returns the path to the unzipped directory.
    """
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded. Please upload a .zip file containing your MRI slices.")
        return None

    zip_filename = list(uploaded.keys())[0]
    print(f"Uploaded '{zip_filename}'. Unzipping...")

    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            # Extract to a subdirectory to keep things organized
            extract_dir = os.path.splitext(zip_filename)[0] # Uses zip filename as directory name
            zip_ref.extractall(extract_dir)
            print(f"Successfully unzipped to '{extract_dir}'.")
            return extract_dir
    except zipfile.BadZipFile:
        print(f"Error: '{zip_filename}' is not a valid zip file. Please check the file.")
        return None
    except Exception as e:
        print(f"An error occurred during unzipping: {e}")
        return None

# --- Cell 3: Core Functions (Run this after Cell 2) ---

def load_mri_slices(directory_path):
    """
    Loads all TIFF image slices from the specified directory and stacks them into a 3D NumPy array.
    Assumes image files are .tif and are numerically sortable by name.
    These are the RAW MRI slices that will be fed into your segmentation model.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return None

    image_files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith('.tif')])

    if not image_files:
        print(f"No TIFF files found in '{directory_path}'. Please ensure your slices are .tif files and the zip structure is flat or points directly to the slices.")
        return None

    slices = []
    print(f"Loading {len(image_files)} MRI slices...")
    for filename in image_files:
        filepath = os.path.join(directory_path, filename)
        try:
            with Image.open(filepath) as img:
                if img.mode not in ['L', 'I;16', 'I']:
                    img = img.convert('L') # Convert to 8-bit grayscale for simplicity

                slices.append(np.array(img))
        except Exception as e:
            print(f"Could not load '{filename}': {e}")
            continue

    if not slices:
        print("No slices were successfully loaded. Check file permissions or corruption.")
        return None

    first_slice_shape = slices[0].shape
    for i, s in enumerate(slices):
        if s.shape != first_slice_shape:
            print(f"Warning: Slice '{image_files[i]}' has a different shape ({s.shape}) than the first slice ({first_slice_shape}). Skipping this slice.")
            slices[i] = None
    slices = [s for s in slices if s is not None]

    if not slices:
        print("No consistent slices found after shape validation. 3D volume cannot be formed.")
        return None

    volume = np.stack(slices, axis=0)
    print(f"Successfully loaded 3D volume with shape: {volume.shape} (slices, height, width)")
    return volume

def export_to_obj(vertices, faces, filename):
    """
    Exports the given vertices and faces into a .obj mesh file.
    """
    print(f"Exporting mesh to {filename}...")
    try:
        with open(filename, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"Mesh successfully exported to {filename}.")
    except Exception as e:
        print(f"Error exporting OBJ file: {e}")

def reconstruct_and_visualize_3d(volume_data, output_obj_filename):
    """
    Performs 3D surface reconstruction. This function now integrates a placeholder
    for your custom segmentation model.
    """
    if volume_data is None:
        print("Cannot reconstruct: No valid raw volume data provided.")
        return

    segmented_volume = None # Initialize to None

    print("\n--- Integrating Custom Segmentation Model (Your Code Goes Here) ---")
    if your_segmentation_model is not None:
        try:
            # --- MODEL-SPECIFIC PREPROCESSING AND PREDICTION ---
            # IMPORTANT: Adjust this section based on your specific model's input
            # requirements (e.g., normalization, adding batch/channel dimensions).

            # Example for TensorFlow/Keras model:
            # Assumes model expects input like (batch_size, depth, height, width, channels)
            # Normalize to 0-1 range and add batch/channel dimensions
            input_for_model = volume_data.astype(np.float32) / np.max(volume_data)
            input_for_model = np.expand_dims(input_for_model, axis=(0, -1)) # Add batch and channel dimension (1, D, H, W, 1)

            # Perform prediction
            prediction_output = your_segmentation_model.predict(input_for_model)

            # Post-process model output:
            # Assuming a binary segmentation where the last channel is the tumor probability
            # You might need to adjust slicing based on your model's exact output shape.
            segmented_volume = prediction_output[0, ..., 0] # Remove batch and channel dims, get first channel
            print("Custom model prediction successful.")

        except Exception as e:
            print(f"Error during custom model prediction: {e}")
            print("Falling back to simple intensity thresholding.")
            segmented_volume = None # Ensure fallback logic is triggered

    if segmented_volume is None:
        # Fallback if custom model is not loaded or an error occurred during its prediction
        print("No custom model output available or error occurred. Applying simple intensity thresholding.")
        binary_volume = (volume_data > TUMOR_THRESHOLD).astype(bool)
    else:
        # Use the output from your custom model and apply TUMOR_THRESHOLD for binarization
        # This is where your model's output is converted into a clear binary mask (True/False)
        binary_volume = (segmented_volume > TUMOR_THRESHOLD).astype(bool)
        print(f"Model output binarized using TUMOR_THRESHOLD ({TUMOR_THRESHOLD}).")

    print("--- End of Custom Segmentation Model Integration Section ---\n")


    try:
        # Marching cubes is applied to the *binary* output (either from your model or fallback).
        verts, faces, normals, values = measure.marching_cubes(binary_volume, level=0.5)
        print(f"Marching cubes generated {len(verts)} vertices and {len(faces)} faces for the surface.")
    except ValueError as e:
        print(f"Error during marching cubes: {e}. This often means no contiguous region was found above the specified threshold.")
        print("Suggestions:")
        print("  - Ensure your custom segmentation model is correctly integrated and produces valid output.")
        print(f"  - If your model outputs probabilistic values, try adjusting 'TUMOR_THRESHOLD' for binarization (current: {TUMOR_THRESHOLD}).")
        print("  - If your model outputs binary masks (0 or 255), ensure you're binarizing it correctly (e.g., `segmented_volume > 0`) before Marching Cubes.")
        print("  - Verify that the segmented output contains a contiguous region for Marching Cubes to find.")
        print("  - Check the `segmented_volume` (or `binary_volume` before Marching Cubes) to ensure it contains expected values for the tumor region.")
        return

    # --- Export to OBJ ---
    export_to_obj(verts, faces, output_obj_filename)

    # --- Visualize with Matplotlib ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    cmap='viridis', lw=1, alpha=0.8, antialiased=True)

    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Height)')
    ax.set_zlabel('Z (Slice)')
    ax.set_title(f'3D Reconstruction of MRI Volume (Binarized at {TUMOR_THRESHOLD})')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    print("3D visualization complete. A matplotlib window should have appeared with the reconstructed object.")

# --- Cell 4: Main Execution (Run this last) ---
if __name__ == "__main__":
    print("Welcome to MRI 3D Reconstruction in Google Colab!")

    IMAGE_DIRECTORY = None
    if MRI_DATA_SOURCE_PATH:
        print(f"Attempting to load MRI data from Google Drive: {MRI_DATA_SOURCE_PATH}")
        IMAGE_DIRECTORY = MRI_DATA_SOURCE_PATH
    else:
        print("No Google Drive path specified. Please follow the instructions below to upload your data.")
        # Call the upload function. This will prompt a file upload dialog.
        # The returned path will be the directory where your .tif files are unzipped.
        IMAGE_DIRECTORY = upload_and_unzip_mri_data()

    if IMAGE_DIRECTORY:
        # Load the MRI slices into a 3D volume (these are your raw inputs for your model)
        mri_volume = load_mri_slices(IMAGE_DIRECTORY)

        # If the volume was loaded successfully, perform 3D reconstruction and visualize it
        if mri_volume is not None:
            # Pass the raw MRI volume to the reconstruction function, which now expects
            # you to integrate your segmentation model within it.
            reconstruct_and_visualize_3d(mri_volume, OUTPUT_OBJ_FILENAME)
        else:
            print("3D reconstruction cannot proceed due to issues loading the MRI volume.")
    else:
        print("Image directory could not be set. Please ensure you uploaded a valid zip file or provided a correct Google Drive path.")

    # # To download the generated .obj file after execution, you can use:
    # files.download(OUTPUT_OBJ_FILENAME)
