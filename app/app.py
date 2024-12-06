import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import io
from flask import Flask, request, render_template
import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
# وارد کردن کتابخانه‌ها
from transformers import ViTForImageClassification, ViTFeatureExtractor

# تنظیمات Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # حداکثر اندازه فایل: 16 مگابایت

# ایجاد پوشه آپلود در صورت عدم وجود
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# بارگذاری مدل
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_features = self.model.classifier[1].in_features
        num_classes = 10  # تعداد گروه‌ها
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# بارگذاری وزن‌های مدل
model = CustomModel()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # انتقال مدل به دستگاه
state_dict = torch.load('efficientnet_model.pth', map_location=device)
#(state_dict.keys())
# حذف پیشوند "model."
state_dict_ = {f"model.{k}": v for k, v in state_dict.items()}

#print(state_dict_.keys())
# بارگذاری وزن‌ها در مدل
model.load_state_dict(state_dict_)
#model.load_state_dict(torch.load('efficientnet_model.pth', map_location=device))
model.eval()

# تعریف transform‌ها
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# پردازش تصویر و پیش‌بینی
def predict_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = transform(img).unsqueeze(0)  # تبدیل به tensor و افزودن dimension batch
        img = img.to(device)

        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # نمایش درصد‌ها
        probabilities = probabilities.cpu().numpy().flatten()
        print("Probabilities:", probabilities)
        print("Type of probabilities:", type(probabilities))
        print("Shape of probabilities:", probabilities.shape,'\n\n')

        return probabilities
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


# اضافه کردن مدل ViT
def load_vit_model():
    model_path = 'vit_model.pth'
    vit_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=10  # تعداد گروه‌ها
    )
    vit_model.load_state_dict(torch.load(model_path, map_location=device))
    vit_model.to(device)
    vit_model.eval()
    return vit_model

vit_model = load_vit_model()

# پردازش تصویر برای ViT
def preprocess_image_vit(image_bytes):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    inputs = feature_extractor(images=img, return_tensors="pt")
    return inputs["pixel_values"].to(device)

# پیش‌بینی با ViT
def predict_with_vit(image_bytes):
    img_tensor = preprocess_image_vit(image_bytes)
    with torch.no_grad():
        outputs = vit_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probabilities.cpu().numpy().flatten()


@app.route('/')
def index():
    return render_template('index.html')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if 'file' not in request.files or file.filename == '':
        return 'No file selected'
    
    if not allowed_file(file.filename):
        return 'Unsupported file format'

    try:
        # خواندن تصویر
        image_bytes = file.read()
        
        # پیش‌بینی با مدل EfficientNet
        probs_efficientnet = predict_image(image_bytes)
        max_indices = np.argsort(probs_efficientnet)[-2:]  # دو کلاس برتر
        combined_prob = probs_efficientnet[max_indices[0]] + probs_efficientnet[max_indices[1]]
        # تبدیل به مقدار اسکالر
        combined_prob = float(combined_prob)
        # بررسی شرط استفاده از مدل ViT
        #print(combined_prob,float(probs_efficientnet[max_indices[0]]),"\n\n\n")
        if combined_prob > 0.3 and float(probs_efficientnet[max_indices[1]])<=0.85:
            probs_vit = predict_with_vit(image_bytes)
            print(probs_vit)
        else:
            probs_vit = None

        # ذخیره فایل آپلود شده
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.seek(0)
        file.save(upload_path)

        # گروه‌ها
        groups = {
            7: "گیاه خانگی",
            6: "ظروف آشپزخانه",
            9: "وسایل الکتریکی",
            0: "سنگ‌های قیمتی",
            3: "لوازم خیاطی",
            2: "فرش ماشینی",
            4: "مبل",
            8: "میز",
            5: "کفش و دمپایی زنانه",
            1: "مانتو و تونیک"
        }

        # تعیین نتیجه نهایی
        final_probs = probs_vit if probs_vit is not None else probs_efficientnet
        final_class = np.argmax(final_probs)
        final_group = groups[final_class]

        # ارسال نام دسته‌ها به قالب
        labels = [groups[i] for i in range(len(groups))]

        return render_template(
            'result.html',
            image=filename,
            probs_efficientnet=probs_efficientnet.tolist() if probs_efficientnet is not None else [],
            probs_vit=probs_vit.tolist() if probs_vit is not None else None,
            final_group=final_group,
            labels=labels
        )


    except UnidentifiedImageError:
        return 'Cannot identify the image. Please upload a valid image.'
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
