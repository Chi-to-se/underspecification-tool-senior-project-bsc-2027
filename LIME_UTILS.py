import  sys
import  requests
import  argparse
import  os
import  tensorflow                      as tf 
from    tensorflow.keras.preprocessing  import image
import  numpy                           as np
from    lime                            import lime_image
from    skimage.segmentation            import mark_boundaries
import  matplotlib.pyplot               as plt
import  math
from    rich                            import print
from    PIL                             import Image
import  datetime

# GLOBAL CONFIGURATION
# ====================


SEED = 42

# LIME Parameters
NUM_SAMPLES = 1000
NUM_FEATURES = 5
TOP_LABELS = 2
BATCH_SIZE = 64

# Saving Config
_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("results", f"run_{_timestamp}")
COMBNED_GRID_PATH = os.path.join(SAVE_DIR, "combined_results.png")
UNDERSPECDEG_PATH = os.path.join(SAVE_DIR, "underspec_degree_result.txt")


# Visualization Config
FIG_SIZE = (18,6)



# ====================

def parse_arg():
    parser = argparse.ArgumentParser(description="Upload Models and Images")

    #Add parser
    parser.add_argument("--models", type=str, nargs="+", required=True, help="HELP MODELS")
    parser.add_argument("--images", type=str, nargs="+", required=True, help="HELP IMAGES")
    parser.add_argument("--labels", type=str, nargs="+", required=False, help="HELP LABELS")
    return parser.parse_args()

def preprocess_image(img_path, target_size):

    # Load Image from Memory
    img = image.load_img(img_path, target_size=target_size)

    # Format image to Array(Nomalize to 0-1)
    img_array = image.img_to_array(img) / 255

    return img_array

def num_class_call(model):
    
    # Get last Node of the model
    num_classes = model.output_shape[-1]

    # Check is_binary
    is_binary = (num_classes == 1)

    return num_classes, is_binary

def get_lime_mask(explanation, top_label, num_features=NUM_FEATURES):

    # Get Superpixel weights
    local_exp = dict(explanation.local_exp[top_label])
    
    # Ranking   
    features = sorted(local_exp.items(), key=lambda x : x[1],reverse=True)

    # Seperate positive and negative(Slicing)
    pos_features = [f for f,w in features if w > 0][:num_features]
    neg_features = [f for f,w in features if w < 0][-num_features:]

    # Create Mask
    pos_mask = np.isin(explanation.segments, pos_features)
    neg_mask = np.isin(explanation.segments, neg_features)

    return pos_mask, neg_mask

def create_overlay_image(img_array, pos_mask, neg_mask):

    # Grey and Background
    grey_image = np.mean(img_array, axis=-1)
    background = np.stack([grey_image, grey_image, grey_image], axis=-1) * 0.4
    background_copy = background.copy()

    # Postitive(Green) Area
    background_copy[pos_mask] = img_array[pos_mask] * 0.7
    background_copy[pos_mask, 1] = background_copy[pos_mask, 1] + 0.3

    # Negative(Red) Area
    background_copy[neg_mask] = img_array[neg_mask] * 0.7
    background_copy[neg_mask, 0] = background_copy[neg_mask, 0] + 0.3

    # 0-1 Color
    background_copy = np.clip(background_copy, 0, 1)

    # Boundaries(Border)
    overlay_image = mark_boundaries(
        background_copy, 
        pos_mask.astype(int), 
        color=(0, 1, 0),
        mode='inner'
        )
    overlay_image = mark_boundaries(
        overlay_image,
        neg_mask.astype(int),
        color=(1, 0, 0),
        mode='inner'
    )

    return overlay_image

def plot_and_save_comparison(model_name, image_name, explanation, top_label, predicted_label, predicted_probs, overlay_img, ollama_text=None, save_dir=SAVE_DIR):
    temp, _ = explanation.get_image_and_mask(
            top_label,
            positive_only=True,
            num_features=NUM_FEATURES,
            hide_rest=False
        )
    temp_hide, _ = explanation.get_image_and_mask(
            top_label,
            positive_only=True,
            num_features=NUM_FEATURES,
            hide_rest=True
        )

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[2,1])

    axes = [
        fig.add_subplot(gs[0, 0]), # Model Name
        fig.add_subplot(gs[0, 1]), # Original Image
        fig.add_subplot(gs[0, 2]), # Superpixels Overlay
        fig.add_subplot(gs[0, 3])  # Masked Image
    ]

    ax_bottom = fig.add_subplot(gs[1, :])

    # 1. Model Name
    axes[0].text(0.5, 0.5, model_name, fontsize=30, fontweight='bold', ha='center', va='center')
    axes[0].axis("off")

    # 2. OG Image
    floored_predicted_probs = int(predicted_probs * 10000) / 10000
    axes[1].imshow(temp)
    axes[1].set_title(f"Original Image: {predicted_label}={floored_predicted_probs:.4f}", fontsize=12, fontweight='bold')
    axes[1].axis("off")

    # 3. LIME Bound Images
    axes[2].imshow(overlay_img)
    axes[2].set_title("Superpixels Overlay\nPositive (Green) / Negative (Red)", fontsize=12, fontweight='bold', pad=10)
    axes[2].axis("off")

    # 4. What's model see
    axes[3].imshow(temp_hide)
    axes[3].set_title("Masked Image\nWhat the Model Actually 'Sees'", fontsize=12, fontweight='bold', pad=10)
    axes[3].axis("off")

    # 5. LLM Text
    ax_bottom.axis("off")
    display_text = ollama_text if ollama_text else "Analyzing features with Ollama..."
    ax_bottom.text(
        0.5, 0.5, 
        display_text, 
        fontsize=14, 
        ha='center', 
        va='center',
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.2) # ใส่กรอบสีเทาจาง ๆ ให้สวยงาม
    )

    # Saving
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(
        save_dir,
        f"result_{model_name}_{image_name}.png"
    )

    plt.savefig(save_path)
    plt.close(fig)

    return save_path


    

def combine_results_to_grid(saved_image_paths, output_path=COMBNED_GRID_PATH):

    #
    num_images = len(saved_image_paths)
    if num_images == 0:
        print("No images to combine.")
        return
        
    # Load image and set width and height 
    images = [Image.open(path) for path in saved_image_paths]
    img_width, img_height = images[0].size
    
    # Calculate suitable fig size
    num_cols = int(math.ceil(math.sqrt(num_images)))
    num_rows = int(math.ceil(num_images / num_cols))
    
    # Create empty canvas
    canvas_width = img_width * num_cols
    canvas_height = img_height * num_rows
    grid_image = Image.new("RGB", (canvas_width, canvas_height), color="white")
    
    # Put image into it
    for index, img in enumerate(images):
        col_idx = index % num_cols
        row_idx = index // num_cols
        grid_image.paste(img, (col_idx * img_width, row_idx * img_height))
        
    # Saving
    grid_image.save(output_path)
    
    # Show
    plt.figure(figsize=(15, 10))
    plt.imshow(grid_image)
    plt.axis("off")
    plt.title(f"Combined LIME Results ({num_cols}x{num_rows} Grid)", fontsize=16, fontweight="bold")
    plt.show()

def compute_cosine_distance(mask1,mask2,image_name):

    # Flatten
    mask1 = mask1.flatten().astype("float32")
    mask2 = mask2.flatten().astype("float32")

    # Cosine Similarity
    dot_product = np.dot(mask1,mask2)
    magnitude_a = np.linalg.norm(mask1)
    magnitude_b = np.linalg.norm(mask2)
    
    if magnitude_a == 0 or magnitude_b == 0: return 1.0

    cosine_sim = dot_product / (magnitude_a * magnitude_b)

    # Cosine Distance
    cosine_dis = 1 - cosine_sim
    message = f"COSINE DISTANCE {image_name} (image) : {cosine_dis}"
    print(f"[green]COSINE DISTANCE {image_name} (image) :[/green] [cyan]{cosine_dis}[/cyan]")

    return cosine_dis, message


    


    
