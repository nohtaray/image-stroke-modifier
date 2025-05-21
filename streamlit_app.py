import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import io

st.title("STROKE MODIFIER")

uploaded_files = st.file_uploader(
    "画像をアップロードしてください（複数可）",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)


def process_image_orig(image_file, px, make_white_transparent):
    # RGBAで読み込み、透明部分を白背景で埋める
    image_rgba = Image.open(image_file).convert("RGBA")
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
        alpha_channel = 255 - final_gray  # 黒ほど不透明
        rgb_image = np.zeros((final_gray.shape[0], final_gray.shape[1], 3), dtype=np.uint8)
        rgb_image[:, :, :] = final_gray[:, :, None]
        result_rgba = np.dstack([rgb_image, alpha_channel])
        result_img = Image.fromarray(result_rgba, mode="RGBA")
    else:
        result_img = Image.fromarray(final_gray).convert("RGB")

    return image_rgba, result_img

def process_image_only_stroke(image_file, px, black_threshold):
    # RGBAで読み込み、透明部分を白背景で埋める
    image_rgba = Image.open(image_file).convert("RGBA")
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

    result_img = Image.fromarray(final_gray).convert("RGB")
    # Convert to numpy array once
    img_array = np.array(image_rgba)
    # Apply threshold to RGB channels while preserving alpha
    # threshold より黒いところだけを広げる
    threshold = 1-black_threshold
    mask = final_gray < 255 * threshold
    img_array[mask, 0] = final_gray[mask] # Only modify RGB channels
    img_array[mask, 1] = final_gray[mask] # Only modify RGB channels
    img_array[mask, 2] = final_gray[mask] # Only modify RGB channels

    # alpha_channel = np.array(image_rgba)[:, :, 3]
    # alpha_channel[mask] = 0
    # result_img = Image.fromarray(np.dstack([img_array, alpha_channel]))

    img_array[:, :, 3] = np.array(image_rgba)[:, :, 3]
    img_array[mask, 3] = 255 - final_gray[mask]
    result_img = Image.fromarray(img_array)

    return image_rgba, result_img

def process_image_only_stroke2(image_file, px):
    # RGBAで読み込み、透明部分を白背景で埋める
    image_rgba = Image.open(image_file).convert("RGBA")

    img_np = 255 - np.array(image_rgba)

    kernel_size = abs(px)
    if kernel_size == 0:
        processed_np = img_np.copy()
    else:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if px > 0:
            processed_np = cv2.dilate(img_np, kernel, iterations=1)
        else:
            processed_np = cv2.erode(img_np, kernel, iterations=1)

    final_gray = 255 - processed_np
    result_img = Image.fromarray(final_gray, mode="RGBA")

    return image_rgba, result_img


if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        with st.expander(f"画像 {idx + 1}: {file.name}", expanded=True):
            # 個別設定
            px = st.slider(f"変更ピクセル数", min_value=-20, max_value=20, value=4, key=f"px_{idx}")
            # make_transparent = st.checkbox(f"透過出力", value=False, key=f"transparent_{idx}")
            keep_colors = st.checkbox(f"品質を落として色情報を保持する", value=False, key=f"keep_colors_{idx}")

            # 処理
            if keep_colors:
                black_threshold = st.slider(f"黒色のしきい値", min_value=0.0, max_value=1.0, value=0.8, key=f"black_threshold{idx}")
                if px > 0:
                    orig_img, result_img = process_image_only_stroke(file, px, black_threshold)
                    # orig_img, result_img = process_image_only_stroke2(file, px)
                else:
                    orig_img, result_img = process_image_only_stroke2(file, px)
            else:
                orig_img, result_img = process_image_orig(file, px, make_white_transparent=False)

            # 表示
            col1, col2 = st.columns(2)
            with col1:
                st.image(orig_img, caption="元画像", use_column_width=True)
            with col2:
                st.image(result_img, caption=f"処理後画像（{'太く' if px > 0 else '細く' if px < 0 else '変更なし'}）", use_column_width=True)

            # ダウンロード
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button(
                label="処理後画像をダウンロード",
                data=buf.getvalue(),
                file_name=f"{file.name.rsplit('.', 1)[0]}_processed.png",
                mime="image/png",
                key=f"download_{idx}"
            )

