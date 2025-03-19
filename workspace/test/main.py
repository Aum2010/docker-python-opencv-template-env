from ultralytics import YOLO

# โหลดโมเดล YOLOv8 แบบ pretrained
model = YOLO("yolov8n.pt")

# ทดสอบโมเดลกับรูปภาพ
results = model("https://ultralytics.com/images/zidane.jpg")

# บันทึกผลลัพธ์ลงไฟล์
for i, r in enumerate(results):
    r.save(filename=f"output_{i}.jpg")  # เซฟภาพพร้อมกล่องตรวจจับ

print("✅ ผลลัพธ์ถูกบันทึกลงไฟล์แล้ว!")