import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import io

st.title("線を太く／細くするツール（透明PNG対応）")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"])
px = st.slider("変更ピクセル数（マイナスで細く・プラスで太く）", min_value=-20, max_value=20, value=0)

if uploaded_file:
    # RGBAで読み込む
    image_rgba = Image.open(uploaded_file).convert("RGBA")
    image_np = np.array(image_rgba)

    # アルファを使って「透明＝白」として合成（白背景）
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255, 255))
    image_opaque = Image.alpha_composite(background, image_rgba).convert("L")  # モノクロ化

    # 黒背景で反転して加工
    inverted = ImageOps.invert(image_opaque)
    img_np = np.array(inverted)

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
    result_img = Image.fromarray(result_np).convert("RGB")  # 背景は白、透過なしでOK

    # 表示（横並び）
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgba, caption="元画像", use_column_width=True)
    with col2:
        st.image(result_img, caption=f"処理後画像（{'太く' if px > 0 else '細く' if px < 0 else '変更なし'}）", use_column_width=True)

    # ダウンロード
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    st.download_button(
        label="処理後画像をダウンロード",
        data=buf.getvalue(),
        file_name="processed_image.png",
        mime="image/png"
    )

