import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import io

st.title("線を太く／細くするツール")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"])
px = st.slider("変更ピクセル数（マイナスで細く・プラスで太く）", min_value=-20, max_value=20, value=4)

if uploaded_file:
    # 入力画像
    image = Image.open(uploaded_file).convert("L")
    inverted = ImageOps.invert(image)
    img_np = np.array(inverted)

    # カーネル生成
    kernel_size = abs(px)
    if kernel_size == 0:
        processed_np = img_np.copy()
    else:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if px > 0:
            processed_np = cv2.dilate(img_np, kernel, iterations=1)
        else:
            processed_np = cv2.erode(img_np, kernel, iterations=1)

    result_np = 255 - processed_np
    result_img = Image.fromarray(result_np)

    # 横に並べて表示
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="元画像", use_column_width=True)
    with col2:
        st.image(result_img, caption=f"処理後画像（{'太く' if px > 0 else '細く' if px < 0 else '変更なし'}）", use_column_width=True)

    # ダウンロード
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="処理後画像をダウンロード",
        data=byte_im,
        file_name="processed_image.png",
        mime="image/png"
    )

