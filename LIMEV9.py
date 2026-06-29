import  sys
import  tensorflow  as tf
from    lime        import lime_image
import  numpy       as np
from    rich        import print
from    rich.panel  import Panel
from    tqdm        import tqdm
import  os
from    LIME_UTILS  import (
    parse_arg, 
    preprocess_image, 
    num_class_call, 
    get_lime_mask, 
    create_overlay_image, 
    plot_and_save_comparison, 
    combine_results_to_grid,
    compute_cosine_distance,
    SEED,
    TOP_LABELS,
    NUM_SAMPLES,
    BATCH_SIZE,
    UNDERSPECDEG_PATH
)
from    LIME_Ollama import (
    call_llm_for_response,
    MODEL
)
# Check Folder
if not os.path.exists("models"):
    os.mkdir("models")
if not os.path.exists("labels"):
    os.mkdir("labels")
if not os.path.exists("results"):
    os.mkdir("results")
if not os.path.exists("images"):
    os.mkdir("images")

# Initailize 
explainer = lime_image.LimeImageExplainer(random_state=SEED)
saved_image_paths = []
args = parse_arg()
models_pth = args.models
imgs_pth = args.images
labels_pth = args.labels
mem_cache = {}

print("[bold cyan]========================================[/bold cyan]")
print(f"[green]Models Loaded[/green] : {models_pth}")
print(f"[green]Image Loaded[/green] : {imgs_pth}")
print(f"[green]Labels Loaded[/green]: {labels_pth}")
print("[bold cyan]========================================[/bold cyan]")

# Loop
for idx, m_path in enumerate(tqdm(models_pth, desc="Overall Progress")):
    # Load model & find class input
    model = tf.keras.models.load_model(m_path)
    m_shape = model.input_shape
    model_name = m_path.split("/")[-1].replace(".h5", "")

    print(Panel(f"Current Model: [bold yellow]{model_name}[/bold yellow] | Shape: {m_shape[1]}x{m_shape[2]}", style="blue"))

    
    # Check Class
    num_classes, is_binary = num_class_call(model)
    if labels_pth and len(labels_pth) == len(models_pth):
        with open(labels_pth[idx], "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = ["Class 0", "Class 1"] if is_binary else [f"Class {n}" for n in range(num_classes)]

    def predict_fn(images_batch):
        images_batch = np.array(images_batch ,dtype=np.float32)
        # preds = model.predict(images_batch, verbose=0)

        batch_size = BATCH_SIZE
        preds_list = []
        for i in tqdm(range(0, len(images_batch), BATCH_SIZE) , desc="LIME Predicting", leave=False):
            batch = images_batch[i:i+batch_size]
            preds_batch = model(batch, training=False).numpy() 
            preds_list.append(preds_batch)
            
        preds = np.vstack(preds_list)
        return np.hstack([1 - preds, preds]) if is_binary else preds

    # 3. Loop each image
    for img_path in imgs_pth:
        img_path_split = img_path.split("/")[-1]
        image_name = img_path.split("/")[-1].split(".")[0]
        
        # Preprocess
        img_array = preprocess_image(img_path, target_size=(m_shape[1], m_shape[2]))
        
        # Predict
        img_input = np.expand_dims(img_array, axis=0)
        # preds = model.predict(img_input, verbose=0)
        preds = model(img_input, training=False).numpy()
        probs = [1 - preds[0][0], preds[0][0]] if is_binary else preds[0]
        
        # LIME Explainer
        explanation = explainer.explain_instance(
            img_array,
            predict_fn,
            top_labels=TOP_LABELS,
            hide_color=0,
            num_samples=NUM_SAMPLES
        )
        
        # Find Top Label
        top_label = explanation.top_labels[0]
        predicted_label = class_names[top_label]
        predicted_probs = probs[top_label]
        
        # Find po/neg mask
        pos_mask, neg_mask = get_lime_mask(explanation, top_label, num_features=5)
        
        # Store pos mask in mem
        if image_name not in mem_cache:
            mem_cache[image_name] = {}

        mem_cache[image_name][model_name] = pos_mask

        # Create overlay image
        overlay_img = create_overlay_image(img_array, pos_mask, neg_mask)
        
        # create comparison image
        save_path = plot_and_save_comparison(
            model_name, image_name, explanation, 
            top_label, predicted_label, predicted_probs, overlay_img,
            ollama_text="Analyzing with LLMs."
        )

        # Send to LLM(Image captioning)
        print(f"[green]Sending[/green] {img_path_split} to {MODEL}")
        try:
            ollama_result = call_llm_for_response(save_path)
        except Exception as e:
            print(f"[red]LLM Error for {img_path_split}: {e}[/red]")
            ollama_result = f"[LLM Error] {e}"

        # Real Save
        save_path = plot_and_save_comparison(
            model_name, image_name, explanation, 
            top_label, predicted_label, predicted_probs, overlay_img,
            ollama_text=ollama_result
        )
        saved_image_paths.append(save_path)

        


        
    # Clear RAM
    tf.keras.backend.clear_session()

# Create Grid Image
combine_results_to_grid(saved_image_paths)



# Underspec Degree Loop
models_names = []

# Get Model names (Maybe Function)
for m_path in models_pth:
    model_name = m_path.split("/")[-1].replace(".h5", "").strip()
    models_names.append(model_name)

# Create Empty file
with open(UNDERSPECDEG_PATH, "w") as f:
    f.write("Underspecification Degree\n")

# Try Loop
if len(models_names) < 2:
    print("[yellow] Comaparison must be from 2 models. [/yellow]")
else:
    for i in range(len(models_names)):
        for j in range(i + 1, len(models_names)):
            model_a = models_names[i]
            model_b = models_names[j]
            distances = []
            for k in mem_cache:
                mask_a = mem_cache[k][model_a]
                mask_b = mem_cache[k][model_b]

                cosine_dis, message = compute_cosine_distance(mask_a, mask_b,k)
                with open(UNDERSPECDEG_PATH, "a") as f:
                    f.write(f"{message}\n")
                distances.append(cosine_dis)
                
            
            distances = np.mean(distances)
            print(f"[green]Cosine Distance : [/green] {model_a} vs {model_b} : [cyan]{distances}[/cyan]")
            with open(UNDERSPECDEG_PATH, "a") as f:
                    f.write(f"Cosine Distance :  {model_a} vs {model_b} : {distances}\n")




