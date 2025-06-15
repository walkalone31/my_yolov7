#!/bin/bash

# --- 腳本設定變數 ---
# 定義 Docker 映像檔的名稱和標籤
# 請確保這個名稱與您本地建置的 YOLOv7 CPU 映像檔名稱一致
YOLOV7_IMAGE="my_yolov7:v1.0"

# 定義容器的名稱
CONTAINER_NAME="yolov7_cpu_container"

# 定義要掛載到容器內部的數據路徑 (可選，如果你的程式需要存取本地數據集)
# 例如：~/my_yolov7_project/datasets -> /app/yolov7_data
LOCAL_YOLO_PATH="/home/wayne/wayne_workspace/ai/code/yolov7" # <--- 請修改為你實際的數據路徑
CONTAINER_YOLO_PATH="/app/yolov7"

# --- 處理 X11 顯示設定 ---
# 獲取 SSH 客戶端 IP (如果通過 SSH 連接)
CLIENT_IP=$(echo $SSH_CLIENT | awk '{ print $1}')
echo "檢測到的客戶端 IP: $CLIENT_IP"

DOCKER_DISPLAY="" # 初始化 DOCKER_DISPLAY

if [ -n "$CLIENT_IP" ]; then
    echo "透過 SSH 連接，配置 X11 轉發到 $CLIENT_IP"
    xhost + "$CLIENT_IP" # 允許該 IP 連接到 X server
    DOCKER_DISPLAY="-e DISPLAY=$DISPLAY -v $HOME/.Xauthority:/root/.Xauthority:rw"
else
    echo "本地連接，配置本地 X11 轉發"
    xhost +local:root # 允許本地 root 連接到 X server (Docker 容器通常以 root 運行)
    DOCKER_DISPLAY="-e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"
fi

# 檢查 DOCKER_DISPLAY 變數是否正確設定
echo "DOCKER_DISPLAY 設定: $DOCKER_DISPLAY"

# --- 檢查並管理現有容器 ---
echo "--- 檢查並管理現有的 Docker 容器 ---"

EXISTING_CONTAINER_ID=$(sudo docker ps -a -q -f name=^/${CONTAINER_NAME}$)

if [ -n "$EXISTING_CONTAINER_ID" ]; then
    echo "偵測到名為 '$CONTAINER_NAME' 的容器 (ID: $EXISTING_CONTAINER_ID) 存在。"

    RUNNING_CONTAINER_ID=$(sudo docker ps -q -f name=^/${CONTAINER_NAME}$)
    if [ -n "$RUNNING_CONTAINER_ID" ]; then
        echo "容器 '$CONTAINER_NAME' 正在運行中。嘗試進入該容器的 shell 環境。"
        sudo docker exec -it "$CONTAINER_NAME" bash -l
    else
        echo "容器 '$CONTAINER_NAME' 已停止。嘗試啟動並進入該容器的 shell 環境。"
        sudo docker start "$CONTAINER_NAME"
        sudo docker exec -it "$CONTAINER_NAME" bash -l
    fi
else
    echo "未偵測到名為 '$CONTAINER_NAME' 的容器。將建立一個新的容器。"
    echo "--- 建立並啟動新的 Docker 容器 ---"

    # 運行新的 Docker 容器
    sudo docker run --name "$CONTAINER_NAME" -it \
        --rm \
        --ipc=host \
        --privileged \
        --device /dev/video0:/dev/video0 \
        --group-add $(getent group video | cut -d: -f3) \
        -v ~/.vim:/root/.vim \
        -v ~/.vimrc:/root/.vimrc \
        -v "$LOCAL_YOLO_PATH:$CONTAINER_YOLO_PATH" \
        $DOCKER_DISPLAY \
        -v /usr/bin/xauth:/usr/bin/xauth \
        "$YOLOV7_IMAGE" \
        # "git config --global --add safe.directory /app/yolov7"
        # bash -l
        bash -c "git config --global --add safe.directory /app/yolov7 && cd /app/yolov7 && bash -l"
fi

echo "--- 腳本執行完畢 ---"


