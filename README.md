# YOLOv7-CPU-Camera-Inference-Dockerized

這個專案提供了一個 Dockerized 的解決方案，用於在 **純 CPU 環境** 下運行 YOLOv7 模型，並透過攝像頭進行實時目標檢測。它基於官方的 [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) 倉庫。

---

## 專案概述

本專案主要目標是提供一個基於 Docker 的解決方案，讓你能夠在沒有獨立 GPU 的筆記型電腦或其他 CPU 環境中運行 YOLOv7 模型，並透過電腦的攝像頭進行實時目標檢測。


```
## 專案結構
.
├── docker/
│   ├── docker_build_yolov7.sh    # 輔助 Docker 映像檔構建的腳本
│   ├── Dockerfile-cpu            # CPU 環境的 Dockerfile 定義
│   └── docker_run_yolov7.sh      # 運行 Docker 容器並配置相機/顯示的腳本
├── .git/                         # Git 倉庫相關文件
├── .gitignore                    # Git 忽略規則文件
└── yolov7/                       # 克隆自 WongKinYiu/yolov7 的核心程式碼
   ├── cfg/
   ├── data/
   ├── detect.py                 # YOLOv7 官方檢測腳本
   ├── models/
   │   └── experimental.py       # (已修改以兼容新版 PyTorch)
   ├── my_code/
   │   ├── cv_cam.py             # 整合攝像頭和 YOLOv7 推論的實時檢測腳本
   │   ├── detect.py
   │   ├── models/
   │   ├── utils/
   │   ├── weights/         
├── utils/                    # YOLOv7 官方工具函數
├── weights/                  # 建議用於存放模型權重的目錄 (需要手動創建和下載)
└── yolov7.pt                 # 你的 YOLOv7 預訓練模型權重 (請確保已下載)
└── ... (其他 YOLOv7 官方文件)
```


## 主要新增和修改

1.  **CPU 環境的 Dockerfile (`docker/Dockerfile-cpu`)**
    * 基於 `python:3.9-buster` 完整版映像檔。
    * 安裝了所有必要的系統級依賴，確保 **OpenCV 的 GUI 顯示 (`cv2.imshow()`) 和攝像頭視訊流 (`/dev/videoX`) 能夠在容器中正常工作**。這包括 Qt5、GTK+ 3、VTK7 以及各種圖像和視訊編解碼庫（如 FFmpeg 和 GStreamer）。
    * 配置了 **CPU 版的 PyTorch** 安裝。

2.  **實時攝像頭推論腳本 (`yolov7/my_code/cv_cam.py`)**
    * 整合了攝像頭影像擷取與 YOLOv7 模型推論功能，實現實時目標檢測。
    * 腳本會**自動尋找可用的攝像頭索引**。
    * 已修正 YOLOv7 載入權重時可能發生的 `_pickle.UnpicklingError` (通過修改 `models/experimental.py` 中的 `torch.load` 參數)。
    * 已解決 `cv_cam.py` 中因變數作用域導致的 `NameError`。

3.  **Docker 輔助腳本 (`docker/docker_build_yolov7.sh` 和 `docker/docker_run_yolov7.sh`)**
    * `docker_build_yolov7.sh`: 用於簡化 Docker 映像檔的構建過程。
    * `docker_run_yolov7.sh`: 自動處理 **X11 顯示轉發**、**攝像頭設備掛載**（例如 `/dev/video0`）、**權限配置**，並將**整個 `yolov7` 專案資料夾從主機掛載到容器內部**。同時，它也自動處理了 Git 倉庫在 Docker 掛載點中可能出現的「可疑所有權 (dubious ownership)」問題。

## 環境要求

* **Docker Desktop** (Windows/macOS) 或 **Docker Engine** (Linux)
* **Linux 作業系統** (推薦，特別是對於相機和 X11 轉發，本專案主要在此環境下測試)
* **CPU** (本專案專為 CPU 環境優化，無需獨立 GPU)
* 一個可用的 **USB 網路攝影機** 或 **內建攝像頭**


## 使用方法

### 1. 克隆本倉庫

首先，將本倉庫克隆到你的本地機器上：


假設你希望將專案克隆到 ``~/wayne_workspace/ai/code/``
```bash
cd ~/wayne_workspace/ai/code/
git clone <你的Git倉庫URL> . # 克隆到當前目錄
```
注意： 確保你將倉庫克隆到一個你擁有權限的目錄，以避免 Git 相關的權限問題。

### 2. 下載 YOLOv7 模型權重

下載你需要的 YOLOv7 預訓練模型
例如 yolov7.pt 或 yolov7-tiny.pt），並將它們放入 ``yolov7/weights/`` 資料夾中。如果 ``weights/`` 資料夾不存在，請手動創建。
```Bash
# 在你專案的根目錄下 (即包含 docker/ 和 yolov7/ 的目錄)
cd yolov7/ # 進入yolov7子目錄
mkdir -p weights/
wget -c [https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) -P weights/
wget -c [https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt) -P weights/
```
下載更多模型請參考 YOLOv7 官方 GitHub 發布頁面

### 3. 修改 YOLOv7 源碼 (一次性)
為了兼容新版 PyTorch 的模型載入安全特性，你需要修改 YOLOv7 原始碼中的一個文件。
# 從你的專案根目錄進入 yolov7 子目錄
```Bash
cd yolov7/

# 使用你喜歡的編輯器打開
nano models/experimental.py # 或者 vi models/experimental.py

# 找到這行或類似的:
# ckpt = torch.load(w, map_location=map_location)
# 修改為:
# ckpt = torch.load(w, map_location=map_location, weights_only=False)
```
保存並退出編輯器。

### 4. 構建 Docker 映像檔
進入 ``docker/`` 目錄，然後執行構建腳本：
```Bash
cd ~/wayne_workspace/ai/code/docker/
chmod +x docker_build_yolov7.sh # 賦予執行權限
./docker_build_yolov7.sh
```
docker_build_yolov7.sh 內部會執行 ``docker build -f Dockerfile-cpu -t my_yolov7:v1.0 .``

### 5. 運行 Docker 容器並啟動實時檢測
在運行前，請編輯 docker/docker_run_yolov7.sh 腳本：

確認 YOLOV7_IMAGE： 確保 ``YOLOV7_IMAGE="my_yolov7:v1.0"`` 與你構建的映像檔名稱一致。
確認 LOCAL_YOLO_PATH： 請務必將此變數設定為你本地電腦上 yolov7/ 資料夾的絕對路徑。
例如： ``/home/wayne/wayne_workspace/ai/code/yolov7``
確認攝像頭設備： 腳本預設掛載 ``/dev/video0`` 如果你的攝像頭是其他設備（例如 /dev/video1），請修改腳本中的 ``--device`` 參數。
你可以使用 ``ls -l /dev/video*`` 和 ``v4l2-ctl --list-devices`` 在主機上確認。

執行運行腳本：
```Bash
cd ~/wayne_workspace/ai/code/docker/
chmod a+x docker_run_yolov7.sh # 賦予執行權限
./docker_run_yolov7.sh
```
一旦容器啟動並進入 bash shell 環境，在容器內部執行以下命令來啟動攝像頭實時檢測：
```Bash
# 確保你當前在 /app/yolov7 目錄下（這是容器內掛載的 YOLOv7 專案根目錄）
cd /app/yolov7 

# 運行實時檢測腳本 (假設 cv_cam.py 在 my_code/ 子資料夾中)
python my_code/cv_cam.py 
```
程式將會自動尋找攝像頭並顯示實時檢測結果。按下 ``q`` 鍵退出。

