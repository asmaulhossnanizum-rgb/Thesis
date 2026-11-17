# Vision-Based Detection of Traffic Congestion & Roadside Encroachment in Urban Bangladesh Using Deep Learning Techniques

B.Sc. thesis, Department of Computer Science and Engineering, Port City International University, Chattogram, Bangladesh (2026).

| | |
|---|---|
| **Authors** | Asmaul Hossna Nizum (ID: CSE 028 076 07), Sharmin Akter (ID: CSE 029 077 17) |
| **Supervisor** | Nusratul Jannat, Lecturer, Department of CSE, Port City International University |
| **Dataset** | TFP-BD (Bangladeshi Traffic Flow Dataset), Mendeley Data: https://data.mendeley.com/datasets/h8bfgtdp2r/6 |
| **Dataset paper** | M. M. Islam et al., "TFP-BD: An Image Dataset for Traffic Flow and Pedestrian Movement Analysis on Bangladeshi Urban Roads," *Data in Brief*, 2025. https://doi.org/10.1016/j.dib.2025.111398 |

## What this project does

Five object detectors (YOLOv8n, YOLO11n, RT-DETR-L, Faster R-CNN with a ResNet50-FPN v2 backbone, and EfficientDet-D3) are trained on the TFP-BD dataset (23,678 annotated images, nine classes). Their detections feed two decisions for every image:

- **Traffic congestion (Low / Medium / High):** each detected vehicle is weighted by its Passenger Car Unit (PCU) value, the weights are summed per frame, and the total is compared with two thresholds taken from the validation set (8.45 and 15.80).
- **Roadside encroachment (Clear / Encroached):** a footpath polygon is drawn once per camera and aligned to every frame with SIFT keypoints and a homography. A frame is Encroached when the bottom-centre point of a detected object falls inside the footpath zone.

Two classifiers (ResNet50 and MobileNetV3-Large) are also trained for vehicle-type classification on cropped boxes, and as whole-frame baselines for both decisions.

All notebooks were run on Kaggle (GPU T4 x2).

## Repository contents

All notebooks are in the top folder. The letter or number at the start of each name shows its group and run order.

| Files | Contents |
|---|---|
| `A_Data_Preparation` | Converts the original annotations to YOLO format and creates the 70/20/10 split (16,574 train / 4,736 validation / 2,368 test images, seed 42). |
| `B1` – `B6` | Training of YOLOv8n and YOLO11n, RT-DETR-L, Faster R-CNN, EfficientDet-D3, ResNet50 and MobileNetV3. |
| `C1` – `C8` | PCU-based congestion estimation for each model. |
| `D1` – `D6` | Final evaluation on the 70/20/10 split: detection (mAP), encroachment, congestion, McNemar tests, vehicle-type classifiers, frame classifiers, and the output pictures used in the thesis. |
| `01` – `08` (names ending in `_8010`) | Replication on an 80/10/10 split (18,942 / 2,368 / 2,368) with the **same 2,368 test images**: retraining of all models and the same evaluation. |

### Run order

`A` → `B1`–`B6` → `C1`–`C8` → `D1`–`D6` → `01`–`08` (80/10/10).

The evaluation notebooks read the dataset and the outputs (weights, split lists) of the earlier notebooks as Kaggle inputs; the first cell of each notebook lists what it needs.

## Main results (70/20/10 split, 2,368 test images)

**Object detection**

| Model | mAP@0.5 | mAP@0.75 | mAP@0.5–0.95 |
|---|---:|---:|---:|
| RT-DETR-L | 73.1 | 60.9 | 50.7 |
| YOLOv8n | 73.2 | 60.5 | 50.4 |
| YOLO11n | 71.0 | 57.9 | 48.4 |
| Faster R-CNN | 71.1 | 53.8 | 46.5 |
| EfficientDet-D3 | 66.5 | 50.6 | 43.6 |

**PCU-based congestion classification (percent)**

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| YOLOv8n | 74.96 | 75.45 |
| YOLO11n | 74.37 | 74.88 |
| EfficientDet-D3 | 69.81 | 69.31 |
| Faster R-CNN | 63.72 | 62.16 |
| RT-DETR-L | 42.06 | 34.90 |

**Roadside encroachment (1,605 aligned test frames)**

| Model | Agreement | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| YOLOv8n | 86.7% | 84.9% | 76.1% | 80.2% |
| YOLO11n | 86.8% | 83.2% | 78.5% | 80.8% |
| RT-DETR-L | 86.6% | 83.9% | 76.9% | 80.3% |
| Faster R-CNN | 86.4% | 77.7% | 86.4% | 81.8% |
| EfficientDet-D3 | 86.5% | 80.0% | 82.4% | 81.2% |

McNemar's exact test found no significant difference between any pair of detectors for encroachment.

**Vehicle-type classification (26,740 test crops):** ResNet50 95.3% accuracy (macro-F1 81.6%), MobileNetV3-Large 95.1% accuracy (macro-F1 81.1%) with about one-sixth of the parameters.

## Main tools

Python, PyTorch, TorchVision, Ultralytics (YOLOv8, YOLO11, RT-DETR), effdet, OpenCV, scikit-learn, NumPy, Pandas, Matplotlib.
