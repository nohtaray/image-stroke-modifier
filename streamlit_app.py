import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import io

st.title("STROKE MODIFIER")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"])
px = st.slider("変更ピクセル数（マイナスで細く・プラスで太く）", min_value=-20, max_value=20, value=0)
make_white_transparent = st.checkbox("透過出力", value=True)

if uploaded_file:
    # RGBAで読み込み、透明部分を白背景で埋める
    image_rgba = Image.open(uploaded_file).convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255, 255))
    image_opaque = Image.alpha_composite(background, image_rgba).convert("L")  # 白背景で合成しモノクロに

    # 線画処理（反転→膨張or収縮→再反転）
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

    final_gray = 255 - processed_np  # 最終グレースケール画像（線が黒）

    if make_white_transparent:
        # アルファチャンネルを白さで作成（白=255, 黒=0）
        alpha_channel = 255 - final_gray
        rgb_image = np.zeros((final_gray.shape[0], final_gray.shape[1], 3), dtype=np.uint8)
        rgb_image[:, :, :] = final_gray[:, :, None]  # グレースケール→RGB
        result_rgba = np.dstack([rgb_image, alpha_channel])
        result_img = Image.fromarray(result_rgba, mode="RGBA")
    else:
        # 白背景のまま出力
        result_img = Image.fromarray(final_gray).convert("RGB")

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

