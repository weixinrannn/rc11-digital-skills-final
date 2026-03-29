import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ==============================
# 1. Model Paths
# ==============================
yolo_model_path = r"C:\Users\86130\runs\detect\train2\weights\best.pt"
sam_checkpoint = r"D:\sam_vit_h_4b8939.pth"

# ==============================
# 2. Input Folder
# ==============================
input_folder = r"D:\00\paintings_removed_review"

# ==============================
# 3. Output Folders
# ==============================
output_vis = r"D:\yolo_output_visual"
output_labels = r"D:\yolo_output_labels"
output_crops = r"D:\yolo_output_crops"
output_sam = r"D:\sam_fragments"

os.makedirs(output_vis, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)
os.makedirs(output_crops, exist_ok=True)
os.makedirs(output_sam, exist_ok=True)

# ==============================
# 4. Load Models
# ==============================
yolo_model = YOLO(yolo_model_path)

sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

window_count = 0
sam_count = 0

# ==============================
# 5. Process Images
# ==============================
for filename in os.listdir(input_folder):

    if filename.lower().endswith((".jpg", ".png", ".jpeg")):

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            continue

        height, width = image.shape[:2]

        # ===== YOLO Detection =====
        results = yolo_model(image_path, conf=0.25)

        label_lines = []

        # ===== Set SAM image =====
        predictor.set_image(image)

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(int, box)

                # ===== 1. Visualization =====
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label_text = f"window {int(scores[i]*100)}%"
                cv2.putText(image, label_text, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # ===== 2. YOLO Label =====
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height

                label_lines.append(f"0 {x_center} {y_center} {bw} {bh}")

                # ===== 3. Crop Window =====
                crop = image[y1:y2, x1:x2]

                if crop.shape[0] > 30 and crop.shape[1] > 30:
                    crop_path = os.path.join(output_crops, f"window_{window_count}.jpg")
                    cv2.imwrite(crop_path, crop)
                    window_count += 1

                # ===== 4. SAM Segmentation =====
                input_box = np.array([x1, y1, x2, y2])

                masks, scores_sam, logits = predictor.predict(
                    box=input_box,
                    multimask_output=False
                )

                mask = masks[0]

                # Apply mask (transparent background)
                sam_image = image.copy()
                sam_image[~mask] = 0

                sam_path = os.path.join(output_sam, f"sam_{sam_count}.png")
                cv2.imwrite(sam_path, sam_image)
                sam_count += 1

        # ===== Save Visualization =====
        vis_path = os.path.join(output_vis, filename)
        cv2.imwrite(vis_path, image)

        # ===== Save Labels =====
        txt_name = filename.rsplit(".", 1)[0] + ".txt"
        txt_path = os.path.join(output_labels, txt_name)

        with open(txt_path, "w") as f:
            f.write("\n".join(label_lines))

# ==============================
# Done
# ==============================
print("✅ Process Completed")
print("Total YOLO Windows:", window_count)
print("Total SAM Fragments:", sam_count)
