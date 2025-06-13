# Thyroid-Nodule-Detection-and-Classification-System
此專案為一套 甲狀腺超音波影像分析系統，能夠：

1. 使用 YOLO 模型偵測甲狀腺結節位置
1. 擷取 ROI 區域並使用深度學習模型進行良/惡性分類
1. 透過簡易 Web GUI 提供使用者上傳影像並即時獲得診斷結果

## 🔧 功能總覽
| 功能          | 說明                                   |
| ----------- | ------------------------------------ |
| YOLOv8 結節偵測 | 輸入原始影像，輸出是否有結節並框出位置                  |
| 圖像分類        | 使用 `regnety_004` 模型對結節進行良/惡性分類       |
| 圖形介面        | 可上傳影像並顯示原圖與 YOLO 框選圖，以及分類結果與信心值      |
| 訓練腳本        | 提供多模型訓練流程與驗證圖表（ROC、Confusion Matrix） |

## 📁 資料集結構與分割方式
本系統採用分層資料夾結構，並使用以下比例切分資料：
* `test`：原始資料中的 20%（保留為最終測試集）
* `train`：其餘資料中的 80% × 80%（= 64%）作為訓練集
* `val`：其餘資料中的 80% × 20%（= 16%）作為驗證集

### YOLO YAML 設定範例：
```yaml!
train: C:/temp/YOLO_DATA_aug/images/train
val: C:/temp/YOLO_DATA_aug/images/val
test: C:/temp/YOLO_DATA_aug/images/test

nc: 1
names: ["Nodule"]
```

## 🧪 模型訓練方式
### 🔍 YOLOv8 結節偵測
使用 `Ultralytics YOLO` 套件訓練，預測是否含有結節：
```bash!
yolo detect train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
```
### 🔬 圖像分類（良性 vs 惡性）
使用 `timm` 套件進行訓練：
```bash!
python THY_classification.py
```
支援多模型比較，包括：
* resnet50
* efficientnet_b0
* vit_base_resnet50d_224
* swinv2_small_window8_256
* regnety_004 ✅（本專案最佳表現）

訓練結束會自動儲存最佳模型與以下評估：
* Accuracy
* Precision / Recall / F1-score
* ROC AUC
* 混淆矩陣圖與 ROC 曲線


## 🖥️ 使用 Web GUI
啟動網頁後，使用者可上傳任一張原始影像，後端流程如下：
1. 使用 YOLO 模型偵測是否含有結節
1. 若無結節 → 顯示「甲狀腺無明顯異常」
1. 若有結節 → 擷取 ROI，輸入至分類模型
1. 回傳預測類別與信心值，同時顯示：
    * 原始影像
    * YOLO 框選圖

### ✅ 啟動方式
```bash!
pip install -r requirements.txt
python app.py
```
開啟網址：`http://127.0.0.1:5000`


📦 套件需求（requirements.txt）
```txt!
flask
torch
torchvision
timm
ultralytics
opencv-python
pillow
matplotlib
scikit-learn
```

### 📁 專案結構範例
```bash!
├── app.py                      # 主網頁後端
├── THY_classification.py       # 多模型訓練腳本（良/惡性分類）
├── best.pt                     # 訓練好的 YOLO 權重（結節偵測）
├── regnety_004_best.pth        # 訓練好的分類模型（regnety_004）
├── data.yaml                   # YOLO 訓練用設定檔（train/val/test 位置與類別）
├── static/
│   ├── uploads/                # 使用者上傳的圖片
│   └── results/                # YOLO 預測圖（含框）
├── templates/
│   ├── index.html              # 上傳頁面（Bootstrap 美化）
│   └── result.html             # 顯示預測圖與文字
└── README.md                   # 本說明文件
```
