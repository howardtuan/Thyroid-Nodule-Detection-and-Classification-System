import os
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm
from ultralytics import YOLO

# -------------------- 設定區 --------------------
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['benign', 'malignant']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 載入模型
classification_model_path = r"C:\\temp\\MyGUI\\regnety_004_best.pth"
detection_model_path = "best.pt"  # 你訓練好的 YOLO 權重

cls_model = timm.create_model("regnety_004", pretrained=False, num_classes=2)
cls_model.load_state_dict(torch.load(classification_model_path, map_location='cpu'))
cls_model.eval()

yolo_model = YOLO(detection_model_path)

# -------------------- Flask App --------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


def predict_roi(roi_img):
    img_pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        output = cls_model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()
    return CLASS_NAMES[pred_idx], confidence


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            # YOLO 預測
            results = yolo_model.predict(source=img_path, save=False, conf=0.3)
            boxes = results[0].boxes

            img_bgr = cv2.imread(img_path)
            yolo_out = img_bgr.copy()

            if boxes is None or len(boxes) == 0:
                message = "甲狀腺無明顯異常"
                return render_template('result.html',
                                       orig_image=url_for('static', filename=f'uploads/{file.filename}'),
                                       message=message,
                                       title="分析結果")
            else:
                # 偵測有結果 -> 畫框並擷取 ROI
                x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].tolist())
                cv2.rectangle(yolo_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                roi = img_bgr[y1:y2, x1:x2]

                # 存 yolo 預測圖
                yolo_filename = f'yolo_{file.filename}'
                yolo_path = os.path.join(app.config['RESULT_FOLDER'], yolo_filename)
                cv2.imwrite(yolo_path, yolo_out)

                # 預測分類
                label, prob = predict_roi(roi)
                message = f"預測結果：{label}（信心值 {prob:.2f}）"

                return render_template('result.html',
                                       orig_image=url_for('static', filename=f'uploads/{file.filename}'),
                                       yolo_image=url_for('static', filename=f'results/{yolo_filename}'),
                                       message=message,
                                       title="分析結果")

    return render_template('index.html', title="甲狀腺結節檢測")


if __name__ == '__main__':
    app.run(debug=True)
