import deeplake
import os
import numpy as np
import cv2
from tqdm import tqdm

# Load full 300W dari Activeloop
print("Loading 300W dataset from activeloop...")
ds = deeplake.load("hub://activeloop/300w")

# Buat folder lokal
os.makedirs("300W_FIXED/images", exist_ok=True)
os.makedirs("300W_FIXED/pts", exist_ok=True)

def save_pts_file(landmarks, filename):
    """Save landmarks in .pts format"""
    with open(filename, 'w') as f:
        f.write("version: 1\n")
        f.write("n_points: 68\n")
        f.write("{\n")
        for (x, y) in landmarks:
            f.write(f"{x:.6f} {y:.6f}\n")
        f.write("}\n")

def convert_keypoints_to_68_landmarks(keypoints):
    """
    Convert (204,1) keypoints to (68,2) landmarks
    Format: [x1, y1, z1, x2, y2, z2, ..., x68, y68, z68]
    """
    # Flatten the keypoints
    kp_flat = keypoints.flatten()
    
    # Reshape to (68, 3) - [x, y, z] for each landmark
    kp_xyz = kp_flat.reshape(68, 3)
    
    # Extract only x and y coordinates (ignore z)
    landmarks_xy = kp_xyz[:, :2]  # Take first 2 columns (x, y)
    
    return landmarks_xy.astype(np.float32)

def validate_landmarks(landmarks, img_width, img_height):
    """Validate that landmarks are within image boundaries"""
    # Check if landmarks are within image bounds
    x_valid = (landmarks[:, 0] >= 0) & (landmarks[:, 0] < img_width)
    y_valid = (landmarks[:, 1] >= 0) & (landmarks[:, 1] < img_height)
    
    valid_ratio = np.sum(x_valid & y_valid) / len(landmarks)
    
    # At least 90% of landmarks should be within bounds
    return valid_ratio >= 0.9

# Statistics tracking
total_samples = len(ds)
successful_saves = 0
errors = []

print(f"Total samples in dataset: {total_samples}")
print("Format identified: 68 landmarks × 3 coordinates (x,y,z)")
print("Converting to 68 landmarks × 2 coordinates (x,y)")

# Process all samples
for i in tqdm(range(total_samples), desc="Converting 300W"):
    try:
        # Get image
        img = ds.images[i].numpy()
        h, w = img.shape[:2]
        
        # Get keypoints and convert
        keypoints = ds.keypoints[i].numpy()
        
        # Convert (204,1) to (68,2)
        landmarks = convert_keypoints_to_68_landmarks(keypoints)
        
        # Validate landmarks
        if not validate_landmarks(landmarks, w, h):
            errors.append(f"Index {i}: Landmarks outside image bounds")
            continue
        
        # File paths
        img_path = f"300W_FIXED/images/img_{i:05d}.jpg"
        pts_path = f"300W_FIXED/pts/img_{i:05d}.pts"
        
        # Save image (ensure BGR format for OpenCV)
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        
        success = cv2.imwrite(img_path, img_bgr)
        if not success:
            errors.append(f"Index {i}: Failed to save image")
            continue
        
        # Save landmarks as .pts file
        save_pts_file(landmarks, pts_path)
        
        successful_saves += 1
        
        # Debug info for first few samples
        if i < 3:
            print(f"\nSample {i}:")
            print(f"  Image: {w}×{h}")
            print(f"  Landmarks shape: {landmarks.shape}")
            print(f"  X range: [{landmarks[:,0].min():.1f}, {landmarks[:,0].max():.1f}]")
            print(f"  Y range: [{landmarks[:,1].min():.1f}, {landmarks[:,1].max():.1f}]")
            print(f"  First landmark: ({landmarks[0,0]:.1f}, {landmarks[0,1]:.1f})")
        
    except Exception as e:
        errors.append(f"Index {i}: {str(e)}")
        continue

# Summary
print("\n" + "="*60)
print("CONVERSION SUMMARY")
print("="*60)
print(f"Total samples processed: {total_samples}")
print(f"Successfully converted: {successful_saves}")
print(f"Errors encountered: {len(errors)}")
print(f"Success rate: {successful_saves/total_samples*100:.1f}%")

# Show errors if any
if errors:
    print(f"\nErrors (first 5):")
    for error in errors[:5]:
        print(f"  - {error}")
    
    if len(errors) > 5:
        print(f"  ... and {len(errors) - 5} more errors")

# Verify saved files
saved_images = len([f for f in os.listdir("300W_FIXED/images") if f.endswith('.jpg')])
saved_pts = len([f for f in os.listdir("300W_FIXED/pts") if f.endswith('.pts')])

print(f"\nVerification:")
print(f"Images saved: {saved_images}")
print(f"Landmark files saved: {saved_pts}")
print(f"Files match: {'✅' if saved_images == saved_pts else '❌'}")

if successful_saves > 0:
    print(f"\n✅ Dataset conversion completed successfully!")
    print(f"📁 Saved to: {os.path.abspath('300W_FIXED')}")
    
    # Test a sample to verify format
    test_img_path = "300W_FIXED/images/img_00000.jpg"
    test_pts_path = "300W_FIXED/pts/img_00000.pts"
    
    if os.path.exists(test_img_path) and os.path.exists(test_pts_path):
        print(f"\n🔍 Sample verification:")
        
        # Check image
        test_img = cv2.imread(test_img_path)
        print(f"Sample image shape: {test_img.shape}")
        
        # Check .pts file format
        with open(test_pts_path, 'r') as f:
            pts_lines = f.readlines()
        
        print(f"Sample .pts file:")
        print(f"  - Lines: {len(pts_lines)}")
        print(f"  - Header: {pts_lines[0].strip()}")
        print(f"  - N_points: {pts_lines[1].strip()}")
        print(f"  - First landmark: {pts_lines[3].strip()}")
        print(f"  - Last landmark: {pts_lines[-2].strip()}")
        
        # Parse first landmark to verify format
        try:
            first_landmark = pts_lines[3].strip().split()
            x, y = float(first_landmark[0]), float(first_landmark[1])
            print(f"  - Parsed first landmark: ({x:.1f}, {y:.1f})")
            print(f"  - Format: ✅ Correct")
        except:
            print(f"  - Format: ❌ Error parsing")
    
    print(f"\n🎯 Ready for EAGER training!")
    print(f"Update your dataset_config.yaml:")
    print(f'  path: "{os.path.abspath("300W_FIXED")}"')
    
else:
    print("\n❌ Conversion failed.")

print("="*60)