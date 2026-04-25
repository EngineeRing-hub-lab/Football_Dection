import json
import os
import glob
import cv2
import argparse

def convert_data_YOLO(input_dir, output_dir, split_name):
    folder_pattern = f"football_{split_name}"
    json_paths = sorted(glob.glob(os.path.join(input_dir, folder_pattern, "*/*.json")))
    video_paths = sorted(glob.glob(os.path.join(input_dir, folder_pattern, "*/*.mp4")))
    images_out = os.path.join(output_dir, split_name, "images")
    labels_out = os.path.join(output_dir, split_name, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    for num, (json_file, video_file) in enumerate(zip(json_paths, video_paths)):
        counter = 1
        cap = cv2.VideoCapture(video_file)
        with open(json_file, "r") as file:
            json_data = json.load(file)
        
        width = json_data["images"][0]["width"]
        height = json_data["images"][0]["height"]

        while cap.isOpened():
            flag, frame = cap.read()
            if not flag:
                break
            annotations = [
                item for item in json_data["annotations"] 
                if item.get("image_id") == counter and item.get("category_id", 0) > 2
            ]
            
            file_base_name = f"{split_name}_{num + 1}_{counter}"
            with open(os.path.join(labels_out, f"{file_base_name}.txt"), "w") as f:
                for anno in annotations:
                    bbox = anno["bbox"]
                    xmin, ymin, w, h = bbox
                    x_center = (xmin + (w / 2)) / width
                    y_center = (ymin + (h / 2)) / height
                    w_yolo = w / width
                    h_yolo = h / height
                    cls = 0 if anno.get("category_id") == 3 else 1
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w_yolo:.6f} {h_yolo:.6f}\n")
            cv2.imwrite(os.path.join(images_out, f"{file_base_name}.jpg"), frame)
            counter += 1
def get_args():
    parser = argparse.ArgumentParser(description="Project_Yolov5")
    parser.add_argument(
        "--input", 
        type=str, 
        default="/kaggle/input/datasets/nguyendun/football-detection",
        help="Đường dẫn đến thư mục chứa dataset gốc (có các file .json và .mp4)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="/kaggle/working/football_train",
        help="Thư mục sẽ lưu các file .jpg và .txt sau khi chuyển đổi"
    )
    parser.add_argument("--img-size", type=int, default=3840)
    return parser.parse_args()
if __name__ == "__main__":
    args = get_args()
    # Run for Train
    convert_data_YOLO(args.input, args.output, "train")
    # Run for Validation
    convert_data_YOLO(args.input, args.output, "validation")