import cv2
import numpy as np
import streamlit as st
from skimage import color
import gc  # 引入垃圾回收模块

# ==========================================
# 页面基础设置
# ==========================================
st.set_page_config(page_title="智能图像换色工具-超清无损版", layout="wide")
st.title("🎨 智能图像换色工具 (超清无损版)")
st.markdown("上传模特图、蒙版和参考色，系统将自动推演光影参数并输出原画质结果。")

# ==========================================
# 算法核心函数
# ==========================================
def extract_dominant_lab(img_bgr):
    """使用 K-Means 聚类提取图像中心区域的固有色，避开高光和阴影"""
    h, w = img_bgr.shape[:2]
    crop = img_bgr[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    crop_small = cv2.resize(crop, (100, 100))
    img_lab = cv2.cvtColor(crop_small, cv2.COLOR_BGR2LAB)
    
    pixels = np.float32(img_lab.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3 
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    counts = np.bincount(labels.flatten())
    return centers[np.argmax(counts)]

def get_standard_lab(img_bgr, mask_3d=None):
    img_f = img_bgr.astype(np.float32) / 255.0
    lab_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)
    if mask_3d is not None:
        mask_bool = mask_3d[:, :, 0] > 0.5
        if np.any(mask_bool): 
            res = np.mean(lab_f[mask_bool], axis=0)
            del img_f, lab_f, mask_bool  # 清理内存
            return res
    dominant_lab_8bit = extract_dominant_lab(img_bgr)
    del img_f, lab_f
    return dominant_lab_8bit * [100.0/255.0, 1.0, 1.0] - [0, 128.0, 128.0]

def get_lab_metrics(img_bgr, mask_3d=None):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    if mask_3d is not None:
        mask_bool = mask_3d[:, :, 0] > 0.5
        if np.any(mask_bool): 
            res = np.mean(img_lab[mask_bool], axis=0)
            del img_lab, mask_bool
            return res
    del img_lab
    return extract_dominant_lab(img_bgr)

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
    del blur # 用完即删
    
    img_norm = gray_img.astype(np.float32) / 255.0
    img_gamma = np.power(img_norm + 1e-7, 1.0 / g)
    del img_norm # 用完即删
    
    target_L_val = np.clip(l_t + l_off, 0, 255)
    mask_bool = mask_3d[:, :, 0] > 0.5
    current_mean_l = np.mean(img_gamma[mask_bool]) if np.any(mask_bool) else 0.5
    shift_l = (target_L_val / 255.0) - current_mean_l
    del mask_bool
    
    shadow_map = np.clip(img_gamma + shift_l, 0, 1.0) * 255.0
    del img_gamma # 用完即删
    
    merged_lab = cv2.merge([shadow_map.astype(np.uint8), np.full_like(shadow_map, int(a_t), dtype=np.uint8), np.full_like(shadow_map, int(b_t), dtype=np.uint8)])
    del shadow_map # 用完即删
    
    base_colored_f = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    del merged_lab # 用完即删
    
    mask_low = base_colored_f <= 0.5
    result_f = np.where(mask_low, 2.0 * base_colored_f * high_pass_3d, 1.0 - 2.0 * (1.0 - base_colored_f) * (1.0 - high_pass_3d))
    del mask_low, base_colored_f, high_pass_3d # 用完即删，释放3个大型矩阵
    
    result_bgr_8u = (np.clip(result_f, 0, 1) * 255).astype(np.uint8)
    del result_f 
    
    result_lab = cv2.cvtColor(result_bgr_8u, cv2.COLOR_BGR2LAB)
    del result_bgr_8u 
    
    final_a = np.clip(a_t + a_off, 0, 255)
    final_b = np.clip(b_t + b_off, 0, 255)
    corrected_lab = cv2.merge([result_lab[:, :, 0], np.full_like(result_lab[:, :, 0], int(final_a), dtype=np.uint8), np.full_like(result_lab[:, :, 0], int(final_b), dtype=np.uint8)])
    del result_lab
    
    corrected_bgr_f = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    del corrected_lab
    
    final_f = corrected_bgr_f * mask_3d + (orig_img.astype(np.float32) / 255.0) * (1.0 - mask_3d)
    del corrected_bgr_f
    
    final_res = (np.clip(final_f, 0, 1) * 255.0).astype(np.uint8)
    del final_f
    
    return final_res

def render_neon(orig_img, mask_3d, target_lab, params):
    l_off, ao, bo, dg = params
    l_t, a_t, b_t = target_lab.astype(float)
    
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_detail = (clahe.apply(gray).astype(np.float32) / 255.0)
    
    avg_detail = np.mean(gray_detail[mask_3d[:, :, 0] > 0.1])
    detail_map = gray_detail - avg_detail
    del gray_detail # 用完即删
    
    t_l, t_a, t_b = np.clip(l_t + l_off, 0, 255), np.clip(a_t + ao, 0, 255), np.clip(b_t + bo, 0, 255)
    l_layer = np.clip(np.full(gray.shape, t_l, dtype=np.float32) + (detail_map * dg), 0, 255)
    del detail_map # 用完即删
    
    final_lab = cv2.merge([l_layer.astype(np.uint8), np.full(gray.shape, int(t_a), dtype=np.uint8), np.full(gray.shape, int(t_b), dtype=np.uint8)])
    del l_layer, gray # 用完即删
    
    res_bgr = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
    del final_lab
    
    gaussian = cv2.GaussianBlur(res_bgr, (0, 0), 2)
    res_bgr = cv2.addWeighted(res_bgr, 1.4, gaussian, -0.4, 0)
    del gaussian
    
    final_out = (res_bgr.astype(np.float32) * mask_3d + orig_img.astype(np.float32) * (1.0 - mask_3d))
    del res_bgr
    
    final_res = np.clip(final_out, 0, 255).astype(np.uint8)
    del final_out
    
    return final_res

# ==========================================
# 工具函数
# ==========================================
def load_uploaded_image(uploaded_file, is_mask=False):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        flags = cv2.IMREAD_UNCHANGED if is_mask else cv2.IMREAD_COLOR
        return cv2.imdecode(file_bytes, flags)
    return None

def create_low_res_proxy(img, max_width=800):
    """草图计算专用：将图片缩小以保护内存"""
    if img is None: return None
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

# ==========================================
# 主流程
# ==========================================
col1, col2, col3 = st.columns(3)
with col1:
    orig_file = st.file_uploader("1. 上传模特图 (原图)", type=['jpg', 'jpeg', 'png'])
with col2:
    mask_file = st.file_uploader("2. 上传蒙版图 (PNG/JPG)", type=['jpg', 'jpeg', 'png'])
with col3:
    ref_file = st.file_uploader("3. 上传参考图", type=['jpg', 'jpeg', 'png'])

if orig_file and mask_file and ref_file:
    if st.button("🚀 开始渲染 (原画质输出)", use_container_width=True):
        with st.spinner("正在推演参数并进行高清渲染... (这可能需要一些时间)"):
            
            # 1. 载入原始高清大图 (High-Res)
            orig_hr = load_uploaded_image(orig_file)
            mask_raw_hr = load_uploaded_image(mask_file, is_mask=True)
            ref_img = load_uploaded_image(ref_file)

            # 2. 生成草图 (Low-Res) 用于飞速迭代测算
            orig_lr = create_low_res_proxy(orig_hr, 800)
            mask_raw_lr = create_low_res_proxy(mask_raw_hr, 800)

            gray_lr = cv2.cvtColor(orig_lr, cv2.COLOR_BGR2GRAY)
            mask_3d_lr = preprocess_mask(mask_raw_lr, orig_lr.shape[:2])
            del mask_raw_lr # 释放原始低分辨率蒙版
            
            gray_hr = cv2.cvtColor(orig_hr, cv2.COLOR_BGR2GRAY)
            mask_3d_hr = preprocess_mask(mask_raw_hr, orig_hr.shape[:2])
            del mask_raw_hr # 释放原始高分辨率蒙版

            target_lab_8bit = get_lab_metrics(ref_img)
            target_lab_std = get_standard_lab(ref_img)
            l, a, b = target_lab_8bit.astype(float)
            is_neon = (a > 160 or a < 100) or (b > 160)
            
            # 3. 动态反馈循环 (只用草图跑)
            candidates = []
            l_off, a_off, b_off = 0.0, 0.0, 0.0
            learning_rate = 0.6 
            
            for i in range(15):
                if is_neon:
                    params = (l_off, a_off, b_off, 120)
                    img_lr = render_neon(orig_lr, mask_3d_lr, target_lab_8bit, params)
                else:
                    params = (1.0, l_off, a_off, b_off)
                    img_lr = render_standard(orig_lr, gray_lr, mask_3d_lr, target_lab_8bit, params)

                current_lab_std = get_standard_lab(img_lr, mask_3d_lr)
                de = color.deltaE_ciede2000(target_lab_std, current_lab_std)
                candidates.append({'params': params, 'de': de})

                err_l = target_lab_std[0] - current_lab_std[0]
                err_a = target_lab_std[1] - current_lab_std[1]
                err_b = target_lab_std[2] - current_lab_std[2]
                l_off += (err_l * 2.55) * learning_rate
                a_off += err_a * learning_rate
                b_off += err_b * learning_rate
                
                # 循环内手动释放草图内存并回收
                del img_lr, current_lab_std
                gc.collect()

            candidates.sort(key=lambda x: x['de'])
            
            # 释放不再使用的草图变量
            del orig_lr, gray_lr, mask_3d_lr
            gc.collect()

            # 4. 筛选并执行最终的高清渲染
            valid_candidates = []
            last_de = -1.0
            for c in candidates:
                if len(valid_candidates) >= 5: break
                if abs(c['de'] - last_de) < 0.02: continue
                valid_candidates.append(c)
                last_de = c['de']
            
            st.success(f"✅ 高清渲染完成！以下为您生成了 {len(valid_candidates)} 张候选图：")
            cols = st.columns(len(valid_candidates) + 1)
            
            with cols[-1]:
                st.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), caption="颜色参考图", use_column_width=True)

            for idx, c in enumerate(valid_candidates):
                with cols[idx]:
                    # 执行高清大图渲染
                    if is_neon:
                        final_hr = render_neon(orig_hr, mask_3d_hr, target_lab_8bit, c['params'])
                    else:
                        final_hr = render_standard(orig_hr, gray_hr, mask_3d_hr, target_lab_8bit, c['params'])
                    
                    # 为了防止转码时内存炸裂，在展示前拷贝给 st.image
                    st.image(cv2.cvtColor(final_hr, cv2.COLOR_BGR2RGB), caption=f"色差: {c['de']:.2f}", use_column_width=True)
                    
                    # 导出 100% 画质 JPG
                    is_success, buffer = cv2.imencode(".jpg", final_hr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    st.download_button(
                        label="⬇️ 下载超清 JPG",
                        data=buffer.tobytes(),
                        file_name=f"Color_Result_{idx+1}_dE_{c['de']:.2f}.jpg",
                        mime="image/jpeg",
                        key=f"jpg_{idx}"
                    )
                    del buffer # 释放 JPG buffer
                    
                    # 导出 绝对无损 PNG
                    is_success_p, buffer_p = cv2.imencode(".png", final_hr)
                    st.download_button(
                        label="⬇️ 下载无损 PNG",
                        data=buffer_p.tobytes(),
                        file_name=f"Color_Result_{idx+1}_dE_{c['de']:.2f}.png",
                        mime="image/png",
                        key=f"png_{idx}"
                    )
                    del buffer_p # 释放 PNG buffer
                    
                    # 极其重要：每一张大图生成和按钮渲染完毕后，立刻删除内存中的成品图
                    del final_hr
                    gc.collect() 
            
            # 流程彻底结束后，清空底图大矩阵
            del orig_hr, gray_hr, mask_3d_hr, ref_img
            gc.collect()
