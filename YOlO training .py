from ultralytics import YOLO
import os
import yaml

# Dataset Information
ROBOFLOW_DATASET_URL = "https://universe.roboflow.com/roboflow-universe-projects/safety-vests"

# Configuration
# Update these paths to match YOUR actual folder structure
YOLO_CODES_PATH = r'C:\Users\kckes\OneDrive\Desktop\Safety Vest Demo\Yolo codes'
DATASET_FOLDER_NAME = 'Safety Vests.v14-rf-detr-medium-576x576.yolov11'  # Change if your folder name is different
DATASET_PATH = os.path.join(YOLO_CODES_PATH, DATASET_FOLDER_NAME)
DATA_YAML = os.path.join(DATASET_PATH, 'data.yaml')
MODEL_SIZE = 'yolo11n.pt'  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
EPOCHS = 50  # Reduced for CPU training - increase to 100 if you have time
IMAGE_SIZE = 576
BATCH_SIZE = 4  # Reduced for CPU - was 16
PROJECT_NAME = 'safety_vest_training'
EXPERIMENT_NAME = 'vest_detection_v1'

print("\n⚙️  CONFIGURATION:")
print("=" * 60)
print("🖥️  Training Mode: CPU (slower but works without GPU)")
print(f"📦 Batch Size: {BATCH_SIZE} (reduced for CPU)")
print(f"🔄 Epochs: {EPOCHS} (reduced for faster training)")
print(f"⏱️  Estimated Time: 1-3 hours (depends on CPU)")
print("=" * 60)

def print_dataset_download_instructions():
    """Print instructions for downloading the dataset"""
    print("\n" + "=" * 60)
    print("📥 DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print(f"\nDataset URL: {ROBOFLOW_DATASET_URL}")
    print("\n📋 Steps to download the dataset:")
    print("   1. Go to: https://universe.roboflow.com/roboflow-universe-projects/safety-vests")
    print("   2. Click 'Download' button")
    print("   3. Select format: 'YOLOv11' or 'YOLO'")
    print("   4. Choose 'download zip to computer'")
    print("   5. Extract the zip file")
    print("   6. Place the extracted folder in 'Yolo codes/' directory")
    print("\n📁 Expected folder structure:")
    print("   Yolo codes/")
    print("   └── Safety Vests.v14-rf-detr-medium-576x576.yolov11/")
    print("       ├── data.yaml")
    print("       ├── train/")
    print("       ├── valid/")
    print("       └── test/")
    print("\n" + "=" * 60 + "\n")

def check_dataset():
    """Check if dataset exists and is properly structured"""
    print("=" * 60)
    print("CHECKING DATASET...")
    print("=" * 60)
    print(f"Looking for dataset at: {DATASET_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    print("=" * 60)
    
    # Check for Yolo codes folder
    if not os.path.exists(YOLO_CODES_PATH):
        print(f"❌ ERROR: 'Yolo codes' folder not found at: {YOLO_CODES_PATH}")
        print(f"\n💡 SOLUTION:")
        print(f"   1. Check if the path is correct")
        print(f"   2. Or update YOLO_CODES_PATH in the script to match your actual path")
        print(f"\n   Example: YOLO_CODES_PATH = r'C:\\Your\\Actual\\Path\\Yolo codes'")
        print_dataset_download_instructions()
        return False
    
    print(f"✓ Yolo codes folder found: {YOLO_CODES_PATH}")
    
    # List what's actually in the Yolo codes folder
    if os.path.exists(YOLO_CODES_PATH):
        contents = os.listdir(YOLO_CODES_PATH)
        print(f"\n📂 Contents of 'Yolo codes' folder:")
        for item in contents:
            print(f"   - {item}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ ERROR: Dataset folder not found at: {DATASET_PATH}")
        print(f"\n💡 SOLUTION:")
        if os.path.exists(YOLO_CODES_PATH):
            print(f"   Update DATASET_FOLDER_NAME to match one of the folders above")
            print(f"   Current setting: DATASET_FOLDER_NAME = '{DATASET_FOLDER_NAME}'")
        print_dataset_download_instructions()
        return False
    
    print(f"✓ Dataset folder found: {DATASET_PATH}")
    
    if not os.path.exists(DATA_YAML):
        print(f"❌ ERROR: data.yaml not found at: {DATA_YAML}")
        print(f"\n💡 SOLUTION: Your dataset folder should contain a 'data.yaml' file")
        return False
    
    print(f"✓ data.yaml found: {DATA_YAML}")
    
    # Check for required folders
    required_folders = ['train/images', 'valid/images']
    for folder in required_folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.exists(folder_path):
            num_images = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✓ {folder}: {num_images} images found")
        else:
            print(f"⚠ WARNING: {folder} not found")
    
    # Read and display data.yaml content
    try:
        with open(DATA_YAML, 'r') as f:
            data_config = yaml.safe_load(f)
            print(f"\n📄 Dataset Configuration:")
            print(f"   Classes: {data_config.get('names', 'Not specified')}")
            print(f"   Number of classes: {data_config.get('nc', 'Not specified')}")
    except Exception as e:
        print(f"⚠ Could not read data.yaml: {e}")
    
    print("\n" + "=" * 60)
    return True

def train_model():
    """Train the YOLO model on safety vest dataset"""
    
    # Check dataset first
    if not check_dataset():
        print("\n❌ Dataset check failed. Please fix the issues above before training.")
        return
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60)
    print(f"Model: {MODEL_SIZE}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("=" * 60 + "\n")
    
    try:
        # Load pre-trained YOLO model
        print(f"Loading pre-trained model: {MODEL_SIZE}")
        model = YOLO(MODEL_SIZE)
        
        # Train the model
        print("\n🚀 Training started...")
        print("=" * 60)
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=EXPERIMENT_NAME,
            patience=20,  # Early stopping patience
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            cache=False,  # Set to True if you have enough RAM
            device='cpu',  # Changed to CPU - remove quotes and use 0 if GPU becomes available
            workers=4,  # Reduced workers for CPU
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=False,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            resume=False,
            amp=False,  # Disabled AMP for CPU
            fraction=1.0,
            profile=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=True
        )
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Display training results
        print(f"\n📊 Training Results:")
        print(f"   Best model saved at: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt")
        print(f"   Last model saved at: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/last.pt")
        
        # Validate the best model
        print("\n🔍 Validating best model...")
        best_model = YOLO(f'{PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt')
        metrics = best_model.val()
        
        print(f"\n📈 Validation Metrics:")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        
        print("\n" + "=" * 60)
        print("🎉 MODEL TRAINING AND VALIDATION COMPLETE!")
        print("=" * 60)
        print(f"\n💾 To use your trained model in the detection GUI:")
        print(f"   1. Copy: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt")
        print(f"   2. To: {DATASET_PATH}/weights/best.pt")
        print(f"   3. Or use the 'Browse' button in the GUI to select it")
        print("\n" + "=" * 60)
        
        # Optionally copy the best model to the dataset weights folder
        try:
            weights_dir = os.path.join(DATASET_PATH, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            
            import shutil
            src = f'{PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt'
            dst = os.path.join(weights_dir, 'best.pt')
            shutil.copy2(src, dst)
            print(f"\n✅ Best model automatically copied to: {dst}")
            print("   You can now run the detection GUI!")
        except Exception as e:
            print(f"\n⚠ Could not auto-copy model: {e}")
            print("   Please copy manually as instructed above.")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERROR DURING TRAINING: {e}")
        print("=" * 60)
        print("\n💡 Common Solutions:")
        print("   1. Out of memory: Reduce BATCH_SIZE to 2")
        print("   2. Too slow: Reduce EPOCHS to 30")
        print("   3. Dataset path wrong: Check DATASET_PATH variable")
        print("   4. Missing images: Check train/images and valid/images folders")
        print("   5. If you have GPU: Change device='cpu' to device=0")
        print("=" * 60)
        raise

def test_model(model_path):
    """Test the trained model"""
    print("\n" + "=" * 60)
    print("TESTING MODEL...")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return
    
    try:
        model = YOLO(model_path)
        
        # Test on validation set
        print(f"Running validation on test set...")
        metrics = model.val()
        
        print(f"\n📊 Test Results:")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
        
        print("\n✅ Model testing complete!")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("YOLO SAFETY VEST DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"\n🔗 Dataset Source: {ROBOFLOW_DATASET_URL}")
    print("=" * 60)
    
    # Train the model
    train_model()
    
    # Optionally test the model after training
    best_model_path = f'{PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt'
    if os.path.exists(best_model_path):
        test_model(best_model_path)
    
    print("\n" + "=" * 60)
    print("ALL DONE! 🎉")
    print("=" * 60)