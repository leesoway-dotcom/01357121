import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 檢查 OpenCV 版本
print(f"OpenCV 版本: {cv2.__version__}")

# 2. 讀取圖片
# 請將 'image_name.jpg' 換成你放在同一個資料夾底下的圖片檔名
image_name = 'image_name.jpg' 

image = cv2.imread(image_name)

if image is None:
    print(f"錯誤：找不到圖片檔案 '{image_name}'，請確認圖片是否與此 Notebook 放在同一個資料夾中！")
else:
    # 3. 顯示圖片的維度與形狀
    print(f"height: {image.shape[0]} pixels")
    print(f"width: {image.shape[1]} pixels")
    print(f"channel: {image.shape[2]}")
    print(f"NumPy 陣列形狀: {image.shape}")

    # Jupyter 使用 matplotlib 來顯示圖片（將 BGR 轉為 RGB）
    print("原始圖片 (Original):")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # 4. 灰階轉換與反白處理 (Task 6-1)
    gray_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    inverted_image = cv2.bitwise_not(gray_image)

    print("數位影像反白結果:")
    plt.imshow(inverted_image, cmap='gray')
    plt.axis('off')
    plt.show()
    
    cv2.imwrite("inverted_image.jpg", inverted_image)
    print("已成功儲存反白圖片為 inverted_image.jpg")

    # 5. 影像二值化處理 (Task 6-2)
    mean_val = np.mean(gray_image)
    print(f"圖片平均灰階值: {mean_val}")

    # 利用平均值作為門檻值進行二值化
    binary_image = np.where(gray_image > mean_val, 255, 0).astype(np.uint8)

    print("二值化結果:")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    plt.show()
    
    cv2.imwrite("binary_image.jpg", binary_image)
    print("已成功儲存二值化圖片為 binary_image.jpg")
```[cite: 1, 2, 3, 4, 5]

### 💡 核心差異說明：
1. **圖片讀取**：本地端不需要用 `google.colab.files.upload()`。請直接把你要處理的圖片（例如 `image_name.jpg`）放在跟你的 `.ipynb` 檔案**同一個資料夾目錄底下**，程式就會自動讀取。
2. **圖片顯示**：本地 Jupyter Notebook 不支援 Colab 的 `cv2_imshow`，因此這裡改用 `matplotlib.pyplot` (`plt.imshow`) 來顯示圖片。同時因為 OpenCV 預設讀取是 BGR 顏色通道，顯示彩色圖時需透過 `cv2.cvtColor` 轉為 RGB 顏色才不會跑色。
