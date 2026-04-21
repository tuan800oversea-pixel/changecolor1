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
    # 融合新代码：略微加大羽化半径(5,5)使边缘更平滑自然
    return np.repeat((cv2.GaussianBlur(m, (5, 5), 0).astype(np.float32) / 255.0)[:, :, np.newaxis], 3, axis=2)

# 【核心升级】植入新代码的极高纹理保留渲染器，并保留原有严格的内存回收规范
def render_standard(orig_img, gray_img, mask_3d, target_lab, params):
    g, l_off, a_off, b_off, detail_boost = params
    l_t, a_t, b_t = target_lab.astype(float)
    
    # 1. 极轻度去噪，过滤杂色但保留90%以上原生纹理
    denoised_gray = cv2.bilateralFilter(gray_img, d=5, sigmaColor=20, sigmaSpace=20)
    
    # 2. 增强褶皱与纹理细节
    blur_layer = cv2.GaussianBlur(denoised_gray, (15, 15), 0)
    detail_layer = cv2.subtract(denoised_gray, blur_layer).astype(np.float32)
    detail_layer = detail_layer * detail_boost
    del blur_layer # 用完即删
    
    # 3. 基础色彩映射
    img_norm = denoised_gray.astype(np.float32) / 255.0
    img_gamma = np.power(img_norm + 1e-7, 1.0 / g)
    del img_norm, denoised_gray # 用完即删
    
    target_L_val = np.clip(l_t + l_off, 0, 255)
    mask_bool = mask_3d[:, :, 0] > 0.5
    current_mean_l = np.mean(img_gamma[mask_bool]) if np.any(mask_bool) else 0.5
    shift_l = (target_L_val / 255.0) - current_mean_l
    del mask_bool
    
    # 亮度映射并合并增强后的细节层
    shadow_map = np.clip((img_gamma + shift_l) * 255.0 + detail_layer, 0, 255.0)
    del img_gamma, detail_layer # 用完即删
    
    # 4. 构建最终 LAB 并合并
    final_a = np.clip(a_t + a_off, 0, 255)
    final_b = np.clip(b_t + b_off, 0, 255)
    
    merged_lab = cv2.merge([
        shadow_map.astype(np.uint8), 
        np.full_like(shadow_map, int(final_a), dtype=np.uint8), 
        np.full_like(shadow_map, int(final_b), dtype=np.uint8)
    ])
    del shadow_map
    
    result_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    del merged_lab
    
    # 5. 原图蒙版混合
    final_f = (result_bgr.astype(np.float32) / 255.0) * mask_3d + \
              (orig_img.astype(np.float32) / 255.0) * (1.0 - mask_3d)
    del result_bgr
    
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
    del gray_detail 
    
    t_l, t_a, t_b = np.clip(l_t + l_off, 0, 255), np.clip(a_t + ao, 0, 255), np.clip(b_t + bo, 0, 255)
    l_layer = np.clip(np.full(gray.shape, t_l, dtype=np.float32) + (detail_map * dg), 0, 255)
    del detail_map 
    
    final_lab = cv2.merge([l_layer.astype(np.uint8), np.full(gray.shape, int(t_a), dtype=np.uint8), np.full(gray.shape, int(t_b), dtype=np.uint8)])
    del l_layer, gray 
    
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
    # 【核心升级】支持多选上传参考图，对应新代码逻辑
    ref_files = st.file_uploader("3. 上传参考图 (可多选)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if orig_file and mask_file and ref_files:
    if st.button("🚀 开始渲染 (原画质输出)", use_container_width=True):
        with st.spinner("正在推演参数并进行高清渲染... (这可能需要一些时间)"):
            
            # 1. 载入原始高清大图 (High-Res)
            orig_hr = load_uploaded_image(orig_file)
            mask_raw_hr = load_uploaded_image(mask_file, is_mask=True)
            
            # 【核心升级】融合多张参考图的 LAB 数据
            all_labs_8bit = []
            all_labs_std = []
            for rf in ref_files:
                ref_img_temp = load_uploaded_image(rf)
                if ref_img_temp is not None:
                    all_labs_8bit.append(get_lab_metrics(ref_img_temp))
                    all_labs_std.append(get_standard_lab(ref_img_temp))
            
            target_lab_8bit = np.mean(all_labs_8bit, axis=0)
            target_lab_std = np.mean(all_labs_std, axis=0)

            # 2. 生成草图 (Low-Res) 用于飞速迭代测算
            orig_lr = create_low_res_proxy(orig_hr, 800)
            mask_raw_lr = create_low_res_proxy(mask_raw_hr, 800)

            gray_lr = cv2.cvtColor(orig_lr, cv2.COLOR_BGR2GRAY)
            mask_3d_lr = preprocess_mask(mask_raw_lr, orig_lr.shape[:2])
            del mask_raw_lr 
            
            gray_hr = cv2.cvtColor(orig_hr, cv2.COLOR_BGR2GRAY)
            mask_3d_hr = preprocess_mask(mask_raw_hr, orig_hr.shape[:2])
            del mask_raw_hr 

            l, a, b = target_lab_8bit.astype(float)
            is_neon = (a > 160 or a < 100) or (b > 160)
            
            # 3. 动态反馈循环 (只用草图跑)
            candidates = []
            l_off, a_off, b_off = 0.0, 0.0, 0.0
            learning_rate = 0.5  # 遵循新代码将学习率调整为 0.5
            
            for i in range(12):  # 遵循新代码调整为 12 次迭代
                if is_neon:
                    params = (l_off, a_off, b_off, 120)
                    img_lr = render_neon(orig_lr, mask_3d_lr, target_lab_8bit, params)
                else:
                    # 【核心升级】增加 detail_boost=1.3 参数，追求写实纹理
                    params = (1.0, l_off, a_off, b_off, 1.3)
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
                if abs(c['de'] - last_de) < 0.05: continue # 遵循新代码收紧阈值防重复
                valid_candidates.append(c)
                last_de = c['de']
            
            st.success(f"✅ 高清渲染完成！以下为您生成了 {len(valid_candidates)} 张候选图：")
            cols = st.columns(len(valid_candidates) + 1)
            
            with cols[-1]:
                # 拿第一张参考图做展示位
                display_ref = load_uploaded_image(ref_files[0])
                st.image(cv2.cvtColor(display_ref, cv2.COLOR_BGR2RGB), caption="混合参考主图", use_column_width=True)

            for idx, c in enumerate(valid_candidates):
                with cols[idx]:
                    # 执行高清大图渲染
                    if is_neon:
                        final_hr = render_neon(orig_hr, mask_3d_hr, target_lab_8bit, c['params'])
                    else:
                        final_hr = render_standard(orig_hr, gray_hr, mask_3d_hr, target_lab_8bit, c['params'])
                    
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
                    del buffer 
                    
                    # 导出 绝对无损 PNG
                    is_success_p, buffer_p = cv2.imencode(".png", final_hr)
                    st.download_button(
                        label="⬇️ 下载无损 PNG",
                        data=buffer_p.tobytes(),
                        file_name=f"Color_Result_{idx+1}_dE_{c['de']:.2f}.png",
                        mime="image/png",
                        key=f"png_{idx}"
                    )
                    del buffer_p 
                    
                    # 极其重要：每一张大图生成和按钮渲染完毕后，立刻删除内存中的成品图
                    del final_hr
                    gc.collect() 
            
            # 流程彻底结束后，清空底图大矩阵
            del orig_hr, gray_hr, mask_3d_hr
            gc.collect()
