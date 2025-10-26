import torch
import timm
import numpy as np
import cv2
from PIL import Image
import gradio as gr
from torchvision import transforms
import requests
import re
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 7
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

feature_maps = None
gradients = None

def save_feature_maps(module, input, output):
    global feature_maps
    feature_maps = output

def save_gradients(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load('/content/drive/MyDrive/datasets/model_teeth_V2.pth', map_location=device))
model.to(device)
model.eval()

target_layer = model.conv2d_7b
target_layer.register_forward_hook(save_feature_maps)
target_layer.register_full_backward_hook(save_gradients)

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam_plus(image_tensor, class_idx=None):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor)

    if class_idx is None:
        class_idx = torch.argmax(output).item()

    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward(retain_graph=True)

    grads = gradients[0]
    fmap = feature_maps[0]

    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3

    global_sum = torch.sum(fmap * grads_power_2, dim=[1, 2], keepdim=True)
    eps = 1e-8
    alpha = grads_power_2 / (2 * grads_power_2 + global_sum * grads_power_3 + eps)
    weights = torch.sum(alpha * torch.relu(grads), dim=[1, 2])

    cam = torch.sum(weights[:, None, None] * fmap, dim=0)
    cam = torch.relu(cam).cpu().detach().numpy()
    cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
    cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8)
    return cam, output.softmax(dim=1)[0][class_idx].item(), class_idx

def overlay_gradcam(original, heatmap):
    img = np.array(original).astype(np.float32) / 255.0
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(np.uint8(img * 255), 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))

def query_llm(payload):
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer ..............................."
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

def suggest_with_llm(disease_name, confidence, audience="Patient"):
    disease_info = {
        "CaS": "Canker Sores – Painful ulcers inside the mouth.",
        "CoS": "Cold Sores – Blisters caused by the herpes simplex virus.",
        "Gum": "Gingivostomatitis – Inflammation of the gums and oral mucosa.",
        "MC": "Mouth Cancer – Malignant tumors in the oral cavity.",
        "OC": "Oral Cancer – Cancer affecting lips, tongue, or throat.",
        "OLP": "Oral Lichen Planus – Chronic inflammatory condition.",
        "OT": "Oral Thrush – Fungal infection caused by Candida albicans.",
    }
    disease_desc = disease_info.get(disease_name, disease_name)

    if audience == "Patient":
        prompt = f"""
        You are a helpful dental assistant.
        An image was classified as "{disease_desc}" with confidence {confidence*100:.2f}%.
        Please provide:
        1. What this disease is in simple terms.
        2. Possible reasons why it occurs (layman explanations).
        3. Suggested actions for self-care or when to see a dentist (not medical advice).
        """
    else:
        prompt = f"""
        You are a clinical assistant for dentists.
        An intraoral image was classified as "{disease_desc}" with confidence {confidence*100:.2f}%.
        Please provide:
        1. A concise clinical description of this condition.
        2. Typical etiological factors and pathophysiology.
        3. Key differential diagnosis points.
        4. Evidence-based management or treatment options.
        5. Any systemic implications or comorbidities.
        """

    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528:fireworks-ai",
        "messages": [{"role": "user", "content": prompt}]
    }
    result = query_llm(payload)
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    return "LLM failed to respond."

def classify_image(img, audience):
    image_tensor = transform(img)
    heatmap, confidence, class_idx = generate_gradcam_plus(image_tensor)
    cam_img = overlay_gradcam(img, heatmap)

    explanation_raw = suggest_with_llm(class_names[class_idx], confidence, audience)
    explanation_clean = re.sub(r"<think>.*?</think>", "", explanation_raw, flags=re.DOTALL).strip()

    with torch.no_grad():
        probs = model(image_tensor.unsqueeze(0).to(device)).softmax(dim=1)[0].cpu().numpy()

    plt.figure(figsize=(6, 3))
    plt.bar(class_names, probs, color="skyblue", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.title("Prediction Probabilities")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.savefig("hist.png")
    plt.close()

    label = f"**{class_names[class_idx]}** ({confidence*100:.2f}%)"
    subtitle = f"### Explanation for **{audience}**"
    return cam_img, label, f"{subtitle}\n\n{explanation_clean}", "hist.png"

def clear_all():
    return None, "**Prediction will appear here.**", "**Explanation will appear here after classification.**", None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🦷 Oral Disease Classifier with Grad-CAM++ + LLM (InceptionResNetV2)")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📤 Upload Image")
            audience_input = gr.Dropdown(["Patient", "Dentist"], value="Patient", label="👥 Audience")
            with gr.Row():
                run_button = gr.Button("🔍 Classify", variant="primary")
                clear_button = gr.Button("🧹 Clear", variant="secondary")

        with gr.Column(scale=2):
            with gr.Tab("Results"):
                gradcam_output = gr.Image(label="🖼️ Grad-CAM++ Heatmap")
                prediction_output = gr.Markdown(value="**Prediction will appear here.**")
                histogram_output = gr.Image(label="📊 Prediction Histogram")

            with gr.Tab("Explanation"):
                explanation_output = gr.Markdown(
                    value="**Explanation will appear here after classification.**"
                )

    run_button.click(
        fn=classify_image,
        inputs=[image_input, audience_input],
        outputs=[gradcam_output, prediction_output, explanation_output, histogram_output]
    )

    clear_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[gradcam_output, prediction_output, explanation_output, histogram_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
