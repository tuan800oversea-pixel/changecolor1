import cv2
import numpy as np
import streamlit as st
from skimage import color
from io import BytesIO
from PIL import Image

# ==========================================
# 页面基础设置
# ==========================================
st.set_page_config(page_title="智能图像换色工具", layout="wide")
st.title("🎨 智能图像换色工具")
st.markdown("上传模特图、蒙版和参考颜色图，自动计算色差并生成替换后的效果。")

# ==========================================
# 算法核心函数 (保留你的原逻辑)
# ==========================================
def get_standard_lab(img_bgr, mask_3d=None):
    img_f = img_bgr.astype(np.float32) / 255.0
    lab_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)
    if mask_3d is not None:
        mask_bool = mask_3d[:, :, 0] > 0.5
        if np.any(mask_bool): return np.mean(lab_f[mask_bool], axis=0)
    h, w = img_bgr.shape[:2]
    return np.mean(lab_f[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)], axis=(0, 1))

def get_lab_metrics(img_bgr, mask_3d=None):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    if mask_3d is not None:
        mask_bool = mask_3d[:, :, 0] > 0.5
        if np.any(mask_bool): return np.mean(img_lab[mask_bool], axis=0)
    h, w = img_bgr.shape[:2]
    return np.mean(img_lab[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)], axis=(0, 1))

# 【修改点】不再传入路径，直接传入解码后的 numpy array
def preprocess_mask(m, shape):
    if m is None: return np.zeros((*shape, 3), dtype=np.float32)
    m = m[:, :, 3] if (len(m.shape) == 3 and m.shape[2] == 4) else (
        cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) if len(m.shape) == 3 else m)
    m = cv2.resize(m, (shape[1], shape[0]))
    _, m = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
    if m[0, 0] > 127: m = cv2.bitwise_not(m)
    return np.repeat((cv2.GaussianBlur(m, (3, 3), 0).astype(np.float32) / 255.0)[:, :, np.newaxis], 3, axis=2)

def render_standard(orig_img, gray_img, mask_3d, target_lab, params):
    g, l_off, a_off, b_off = params
    l_t, a_t, b_t = target_lab.astype(float)
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    high_pass_3d = np.repeat((np.clip(gray_img.astype(np.float32) - blur.astype(np.float32) + 128.0, 0, 255) / 255.0)[:, :, np.newaxis], 3, axis=2)
    img_norm = gray_img.astype(np.float32) / 255.0
    img_gamma = np.power(img_norm + 1e-7, 1.0 / g)
    target_L_val = np.clip(l_t + l_off, 0, 255)
    mask_bool = mask_3d[:, :, 0] > 0.5
    current_mean_l = np.mean(img_gamma[mask_bool]) if np.any(mask_bool) else 0.5
    shift_l = (target_L_val / 255.0) - current_mean_l
    shadow_map = np.clip(img_gamma + shift_l, 0, 1.0) * 255.0
    merged_lab = cv2.merge([shadow_map.astype(np.uint8), np.full_like(shadow_map, int(a_t), dtype=np.uint8), np.full_like(shadow_map, int(b_t), dtype=np.uint8)])
    base_colored_f = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    mask_low = base_colored_f <= 0.5
    result_f = np.where(mask_low, 2.0 * base_colored_f * high_pass_3d, 1.0 - 2.0 * (1.0 - base_colored_f) * (1.0 - high_pass_3d))
    result_bgr_8u = (np.clip(result_f, 0, 1) * 255).astype(np.uint8)
    result_lab = cv2.cvtColor(result_bgr_8u, cv2.COLOR_BGR2LAB)
    final_a = np.clip(a_t + a_off, 0, 255)
    final_b = np.clip(b_t + b_off, 0, 255)
    corrected_lab = cv2.merge([result_lab[:, :, 0], np.full_like(result_lab[:, :, 0], int(final_a), dtype=np.uint8), np.full_like(result_lab[:, :, 0], int(final_b), dtype=np.uint8)])
    corrected_bgr_f = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    final_f = corrected_bgr_f * mask_3d + (orig_img.astype(np.float32) / 255.0) * (1.0 - mask_3d)
    return (np.clip(final_f, 0, 1) * 255.0).astype(np.uint8)

def render_neon(orig_img, mask_3d, target_lab, params):
    l_off, ao, bo, dg = params
    l_t, a_t, b_t = target_lab.astype(float)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_detail = (clahe.apply(gray).astype(np.float32) / 255.0)
    avg_detail = np.mean(gray_detail[mask_3d[:, :, 0] > 0.1])
    detail_map = gray_detail - avg_detail
    t_l, t_a, t_b = np.clip(l_t + l_off, 0, 255), np.clip(a_t + ao, 0, 255), np.clip(b_t + bo, 0, 255)
    l_layer = np.clip(np.full(gray.shape, t_l, dtype=np.float32) + (detail_map * dg), 0, 255)
    final_lab = cv2.merge([l_layer.astype(np.uint8), np.full(gray.shape, int(t_a), dtype=np.uint8), np.full(gray.shape, int(t_b), dtype=np.uint8)])
    res_bgr = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(res_bgr, (0, 0), 2)
    res_bgr = cv2.addWeighted(res_bgr, 1.4, gaussian, -0.4, 0)
    final_out = (res_bgr.astype(np.float32) * mask_3d + orig_img.astype(np.float32) * (1.0 - mask_3d))
    return np.clip(final_out, 0, 255).astype(np.uint8)

# ==========================================
# Streamlit 前端交互与工作流
# ==========================================
def load_uploaded_image(uploaded_file, is_mask=False):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        flags = cv2.IMREAD_UNCHANGED if is_mask else cv2.IMREAD_COLOR
        return cv2.imdecode(file_bytes, flags)
    return None

col1, col2, col3 = st.columns(3)
with col1:
    orig_file = st.file_uploader("1. 上传模特图 (原图)", type=['jpg', 'jpeg', 'png'])
with col2:
    mask_file = st.file_uploader("2. 上传蒙版图 (PNG/JPG)", type=['jpg', 'jpeg', 'png'])
with col3:
    ref_file = st.file_uploader("3. 上传参考图 (目标颜色)", type=['jpg', 'jpeg', 'png'])

if orig_file and mask_file and ref_file:
    if st.button("🚀 开始计算并渲染", use_container_width=True):
        with st.spinner("正在进行误差补偿闭环，请稍候..."):
            # 1. 加载图像
            orig = load_uploaded_image(orig_file)
            mask_raw = load_uploaded_image(mask_file, is_mask=True)
            ref_img = load_uploaded_image(ref_file)

            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            mask_3d = preprocess_mask(mask_raw, orig.shape[:2])

            target_lab_8bit = get_lab_metrics(ref_img)
            target_lab_std = get_standard_lab(ref_img)
            l, a, b = target_lab_8bit.astype(float)
            is_neon = (a > 160 or a < 100) or (b > 160)
            
            mode_text = '【荧光色】' if is_neon else '【常规色】'
            st.info(f"🎨 分析完毕 | 模式: {mode_text} | 目标LAB: {target_lab_8bit}")

            candidates = []
            l_off, a_off, b_off = 0.0, 0.0, 0.0
            learning_rate = 0.6 

            # 2. 动态反馈循环
            progress_bar = st.progress(0)
            for i in range(15):
                if is_neon:
                    img = render_neon(orig, mask_3d, target_lab_8bit, (l_off, a_off, b_off, 120))
                else:
                    img = render_standard(orig, gray, mask_3d, target_lab_8bit, (1.0, l_off, a_off, b_off))

                current_lab_std = get_standard_lab(img, mask_3d)
                de = color.deltaE_ciede2000(target_lab_std, current_lab_std)
                candidates.append({'img': img.copy(), 'de': de})

                err_l = target_lab_std[0] - current_lab_std[0]
                err_a = target_lab_std[1] - current_lab_std[1]
                err_b = target_lab_std[2] - current_lab_std[2]
                l_off += (err_l * 2.55) * learning_rate
                a_off += err_a * learning_rate
                b_off += err_b * learning_rate
                progress_bar.progress((i + 1) / 15)

            candidates.sort(key=lambda x: x['de'])

            # 3. 筛选并展示结果
            st.success("✅ 渲染完成！以下是为您生成的 5 张不同质感的候选图：")
            
            saved_count = 0
            last_de = -1.0
            cols = st.columns(5) # 横向显示5张图

            for c in candidates:
                if saved_count >= 5: break
                if abs(c['de'] - last_de) < 0.02: continue

                status = "PASS" if c['de'] < 1.5 else "BEST"
                name = f"{saved_count + 1:03d}_{status}_dE_{c['de']:.2f}.jpg"
                
                # 转换颜色空间用于网页显示 (BGR -> RGB)
                rgb_img = cv2.cvtColor(c['img'], cv2.COLOR_BGR2RGB)
                
                with cols[saved_count]:
                    st.image(rgb_img, caption=f"色差: {c['de']:.2f}", use_column_width=True)
                    
                    # 生成下载按钮
                    is_success, buffer = cv2.imencode(".jpg", c['img'])
                    st.download_button(
                        label="⬇️ 下载此图",
                        data=buffer.tobytes(),
                        file_name=name,
                        mime="image/jpeg",
                        key=name
                    )

                last_de = c['de']
                saved_count += 1
