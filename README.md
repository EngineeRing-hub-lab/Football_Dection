# ⚽ Football Detection with YOLOv5

![YOLOv5](https://img.shields.io/badge/Model-YOLOv5-orange?style=for-the-badge&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-00BFFF?style=for-the-badge&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Dự án cung cấp giải pháp nhận diện và theo dõi các đối tượng trong trận đấu bóng đá (Cầu thủ, Quả bóng) dựa trên kiến trúc **YOLOv5**. Dự án được tối ưu hóa để xử lý dữ liệu từ video và hỗ trợ huấn luyện linh hoạt trên nhiều cấu hình phần cứng.

---

## 📸 Demo & Visualization
![Football Detection Demo](demo/football_demo.gif)

---

## ✨ Tính năng chính (Key Features)
* 🎯 **Object Detection:** Nhận diện chính xác Cầu thủ (Players) và Bóng (Ball).
* 🔄 **Data Converter:** Script tự động chuyển đổi từ dữ liệu JSON/Video sang chuẩn YOLO (`scripts/convert_yolo.py`).
* ⚡ **High Performance:** Hỗ trợ huấn luyện song song trên **2 GPU** sử dụng PyTorch DDP.
* 🌍 **Flexible Deployment:** Tương thích tốt trên Kaggle, Google Colab và máy cá nhân (Local).

---

## 📂 Cấu trúc thư mục (Directory Structure)
```text
.
├── data/                   # File cấu hình dataset (.yaml)
├── scripts/                # Các script xử lý dữ liệu & chuyển đổi
├── notebooks/              # Jupyter Notebooks chạy trên Kaggle/Colab
├── requirements.txt        # Các thư viện cần thiết
└── README.md               # Hướng dẫn dự án
```

---

## 🛠️ Cài đặt (Installation)

1. **Clone repository:**
   ```bash
   git clone [https://github.com/Engineering-hub-lab/Footbal_Dection.git](https://github.com/Engineering-hub-lab/Footbal_Dection.git)
   cd Footbal_Dection
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Hướng dẫn sử dụng

### 1. Xử lý dữ liệu
Sử dụng script để tự động trích xuất frame từ video và chuyển đổi label JSON sang định dạng YOLO:
```bash
python scripts/convert_yolo.py --input "path/to/raw/data" --output "./data/processed"
```

### 2. Huấn luyện (Training)
Tận dụng sức mạnh Multi-GPU (nếu có):
```bash
# Chạy với 2 GPU
python -m torch.distributed.run --nproc_per_node 2 train.py --weights yolov5s.pt --data data/football.yaml --device 0,1

# Chạy với GPU
python train.py --weights yolov5s.pt --data data/football.yaml --device gpu

# Chạy với CPU (Để test code)
python train.py --weights yolov5s.pt --data data/football.yaml --device cpu
```

### 3. Nhận diện (Inference)
```bash
python detect.py --weights path/to/best.pt --source path/to/video.mp4 --device 0
```

---

## 📥 Tải xuống (Downloads)

Do kích thước tệp lớn, trọng số mô hình và bộ dữ liệu được lưu trữ riêng. Bạn có thể tải xuống từ các liên kết bên dưới:

| Thành phần | Liên kết tải xuống | Ghi chú |
| :--- | :--- | :--- |
| **Model Weights (`best.pt`)** | [🔗 Tải xuống Weights](https://drive.google.com/file/d/1ImZJvKmbsOTaUrtYBbKsK0528QhSf41q/view?usp=sharing) | Trọng số đã huấn luyện tốt nhất |
| **Raw Dataset (JSON/MP4)** | [🔗 Tải xuống Dataset gốc](https://drive.google.com/file/d/1cfahORaEJ7Q5EhCNLUxQV-AhgXSMeEWC/view?usp=sharing) | Dữ liệu chưa xử lý (từ Drive) |

---

## 📊 Thông tin tập dữ liệu (Dataset Info)
* **Các lớp (Classes):** `0: Ball`, `1: Player`.


## 👤 Thông tin tác giả
* **Họ và tên:** TẠ DŨNG BÌNH
* **Học vấn:** Sinh viên năm 3 - Học viện Công nghệ Bưu chính Viễn thông (PTIT).
* **Lĩnh vực:** Artificial Intelligence & Software Engineerin.
* **Email:** binhta.006@gmail.com
