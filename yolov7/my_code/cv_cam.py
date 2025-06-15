import cv2
import time
import torch
import torch.backends.cudnn as cudnn # 雖然是CPU，但有時會被依賴
from numpy import random
import numpy as np 

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# --- 配置參數 ---
weights = 'weights/yolov7-tiny.pt' # 模型權重檔案的路徑，相對於 cv_cam.py
conf_thres = 0.25 # 物體置信度閾值
iou_thres = 0.45  # NMS 的 IoU 閾值
img_size = 640    # 模型輸入圖片大小 (YOLOv7 預設 640x640)
local_device = 'cpu' # 明確指定使用 CPU
view_img = True   # 是否顯示結果 (這裡是攝像頭，所以通常為 True)
classes = None
agnostic_nms = False


def open_and_show_camera():
    """
    Attempts to open a camera and display its feed.
    It first tries to find an available camera index.
    """
    print("--- 嘗試尋找可用的相機索引 ---")
    cam_idx = -1 # Initialize with an invalid index
    max_indices_to_check = 5 # Common practice to check a few indices

    for i in range(max_indices_to_check):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            print(f"相機索引 {i} 可用。")
            cap_test.release() # Release the test capture
            cam_idx = i
            break
        else:
            print(f"相機索引 {i} 不可用。")
    
    if cam_idx == -1:
        print("錯誤：未找到任何可用的相機。請檢查相機連接和驅動。")
        return # Exit if no camera is found

    print(f"\n--- 嘗試開啟選定的相機索引: {cam_idx} ---")
    # Open the camera using the identified index.
    # We can explicitly try CAP_V4L2 backend for Linux if needed,
    # but auto-detection is usually fine.
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2) # Explicitly use V4L2 backend for Linux

    if not cap.isOpened():
        print(f"cam {cam_idx} open failed !! 請檢查相機權限或驅動。")
        return # Exit if the selected camera still can't be opened

    print(f"cam {cam_idx} open success !!")


    print("Import model and weights...")

    device = select_device(local_device)
    half = device.type != 'cpu'  # half precision only supported on cuda
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    #imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    try:
        model = attempt_load(weights, map_location=device) # 載入 FP32 模型
        stride = int(model.stride.max()) # 模型步長
        if half:
            model.half() # 轉換到 FP16 (如果是在 GPU 上)

        # 獲取模型中的類別名稱
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names] # 為每個類別生成隨機顏色

        print("模型載入成功！")
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        print("請確認模型檔案是否存在，或其版本與 PyTorch / YOLOv7 程式碼兼容。")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # Optional: Set resolution and FPS for better compatibility with some cameras
    # Some cameras require specific resolutions to stream properly.
    # Try setting common resolutions if you encounter 'cannot read frame' issues.
    # You can comment these out if not needed.
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    # Give the camera a moment to initialize its stream after opening
    time.sleep(1) 

    print("\n--- 正在從相機讀取影像 ---")
    print("按 'q' 鍵退出視窗。")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("無法讀取影像幀，退出。")
            break # Exit the loop if frame cannot be read

        # 將 OpenCV 的 BGR 圖像轉換為 PyTorch 模型所需的 RGB 格式，並調整尺寸和維度
        # 這是 YOLOv7 內部處理圖像的方式，需要遵循
        img = letterbox(frame, img_size, stride=stride)[0] # 調整大小並填充 (來自 utils.datasets)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW (OpenCV to PyTorch format)
        img = np.ascontiguousarray(img) # 確保內存連續
    
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # 添加批次維度
    
        # 推理
        t1 = time_synchronized()
        with torch.no_grad():   # 不計算梯度
            pred = model(img, augment=False)[0] # 執行模型推論
        # t2 = time_synchronized()
    
        # 應用非最大抑制 (NMS)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        # t3 = time_synchronized()
    
        # 處理檢測結果並繪製到原始幀上
        for i, det in enumerate(pred): # det 是單個圖像的檢測結果
            # Rescale boxes from img_size to original frame size
            # 這裡需要將檢測框的座標從模型輸入尺寸縮放回原始相機幀的尺寸
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
    
                # 繪製每個檢測框
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}' # 標籤：類別名稱 + 置信度
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=2) # 在幀上繪製框


        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # 計算並打印 FPS (幀率)
        fps = 1 / (time_synchronized() - t1)
        print(f"FPS: {fps:.2f}")
        
        # Wait for 1ms and check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("檢測到 'q' 鍵，正在退出...")
            break

    # Release the camera resource and destroy all OpenCV windows AFTER the loop ends
    cap.release()
    cv2.destroyAllWindows()
    print("相機資源已釋放，所有視窗已關閉。")

# 需要手動從 YOLOv7 的 utils.datasets 複製或導入 letterbox 函數
# 簡單定義一個版本，但最好從原始文件導入以確保完整性
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while preserving aspect ratio
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimun rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        color = color * 0 + 255
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, (dw, dh)

if __name__ == "__main__":
    open_and_show_camera()
