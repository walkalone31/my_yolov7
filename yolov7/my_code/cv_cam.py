import cv2
import time

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

        # Display the frame
        cv2.imshow('Camera Feed', frame)
        
        # Wait for 1ms and check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("檢測到 'q' 鍵，正在退出...")
            break

    # Release the camera resource and destroy all OpenCV windows AFTER the loop ends
    cap.release()
    cv2.destroyAllWindows()
    print("相機資源已釋放，所有視窗已關閉。")

if __name__ == "__main__":
    open_and_show_camera()
