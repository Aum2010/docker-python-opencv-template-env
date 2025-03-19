from ultralytics import YOLO

# # โหลดโมเดล YOLOv8
# model = YOLO("yolov8n.pt")

# # แปลงโมเดลเป็น ONNX
# model.export(format="onnx" , )

# print("✅ โมเดลถูกแปลงเป็น onnx แล้ว! (ไฟล์: yolov8n.onnx)")

model_name = 'yolov8n' #@param ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
input_width = 1280 #@param {type:"slider", min:32, max:4096, step:32}
input_height = 720 #@param {type:"slider", min:32, max:4096, step:32}
optimize_cpu = False

model = YOLO(f"{model_name}.pt") 
# model.export(format="onnx", imgsz=[input_height,input_width] , optimize=optimize_cpu)
model.export(format='onnx', opset = 12, imgsz =[640,640])

# print("✅ โมเดลถูกแปลงเป็น onnx แล้ว! (ไฟล์: yolov8n.onnx)")