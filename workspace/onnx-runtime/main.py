import onnxruntime as ort
import cv2
import numpy as np

# โหลด ONNX Runtime Session
session = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])

# โหลดภาพและเตรียมข้อมูล
image_path = "zidane.jpg"  # โหลดจากไฟล์ในเครื่อง
image = cv2.imread(image_path)
original_image = image.copy()  # เก็บภาพต้นฉบับไว้

# ปรับขนาดและทำ Normalize
image = cv2.resize(image, (640, 640))
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))  # เปลี่ยนเป็น (C, H, W)
image = np.expand_dims(image, axis=0)  # เพิ่ม Batch Dimension

# รันโมเดล
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image})  # ได้ผลลัพธ์จากโมเดล

# Debug: เช็คโครงสร้างของ output
print(f"Output Shape: {[o.shape for o in output]}")

# ดึงผลลัพธ์ (เอา batch dimension ออก)
output = output[0].squeeze(0)  # รูปแบบ (1, 84, 8400) → (84, 8400)

# แยกค่าออกมา
boxes = output[:4, :].T  # Bounding boxes (8400, 4)
scores = output[4, :]  # Confidence scores (8400,)
class_ids = output[5:, :].argmax(0)  # Class IDs (8400,)

# Debug: ดูค่าที่ได้จากโมเดล
print(f"Boxes shape: {boxes.shape}, Scores shape: {scores.shape}, Class IDs shape: {class_ids.shape}")

# วาดกรอบสี่เหลี่ยมบนภาพ
for i in range(len(boxes)):
    score = float(scores[i])  # แปลงเป็น float
    class_id = int(class_ids[i])  # แปลงเป็น int
    if score > 0.5:  # กรองเฉพาะวัตถุที่มั่นใจมากกว่า 50%
        x1, y1, x2, y2 = map(int, boxes[i])  # ดึงค่า Bounding Box
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_image, f"Class {class_id} ({score:.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# บันทึกผลลัพธ์ลงไฟล์
cv2.imwrite("output.jpg", original_image)

print("✅ ผลลัพธ์ถูกบันทึกลงไฟล์: output.jpg")
