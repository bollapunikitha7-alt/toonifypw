# editor.py — TOONIFY Streamlit App
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime

# ── Filter Functions ──────────────────────────────────────────────────────────
# [Keep all your existing filter functions here - they remain unchanged]

def to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# [All your apply_* functions remain exactly the same]
def apply_ghibli_soft(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    d = int(9 + smooth * 6)
    d = d if d % 2 == 1 else d + 1
    smooth_img = cv2.bilateralFilter(cv_img, d=d, sigmaColor=75 + smooth * 50, sigmaSpace=75 + smooth * 50)
    hsv = cv2.cvtColor(smooth_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + (1 - color_simp) * 0.6), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=max(3, int(9 - line_w * 4)) | 1, C=int(2 + line_w * 6)
    )
    return to_pil(cv2.bitwise_and(result, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))

# [Continue with all your other filter functions...]
def apply_cell_shade(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    k = max(2, int(4 + color_simp * 8))
    data = cv_img.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 8, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(cv_img.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=max(3, int(11 - line_w * 4)) | 1, C=int(3 + line_w * 5))
    return to_pil(cv2.bitwise_and(quantized, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))

def apply_bw_ink(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    k = max(1, int(smooth * 4)) | 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=max(3, int(13 - line_w * 6)) | 1, C=int(4 + line_w * 8))
    detail_layer = cv2.Canny(blur, int(30 + detail * 50), int(80 + detail * 100))
    result = cv2.bitwise_and(edges, cv2.bitwise_not(detail_layer))
    return to_pil(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))

def apply_vector_flat(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    flat = cv2.pyrMeanShiftFiltering(cv_img, sp=int(10 + smooth * 15), sr=int(20 + color_simp * 40))
    k = max(3, int(5 + color_simp * 10))
    data = flat.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5), 6, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(flat.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, int(20 + line_w * 30), int(60 + line_w * 80))
    kernel = np.ones((max(1, int(line_w * 2)), max(1, int(line_w * 2))), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges_inv = cv2.bitwise_not(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    return to_pil(cv2.bitwise_and(quantized, edges_inv))

def apply_pencil_sketch(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blur_size = int(15 + smooth * 20) | 1
    blurred = cv2.GaussianBlur(inverted, (blur_size, blur_size), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    sketch = cv2.equalizeHist(sketch)
    if detail > 0.3:
        edges = cv2.Canny(gray, 50, 150)
        sketch = cv2.addWeighted(sketch, 1, edges, detail * 0.3, 0)
    return to_pil(cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR))

def apply_watercolor(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    for _ in range(3):
        cv_img = cv2.bilateralFilter(cv_img, d=9, sigmaColor=75, sigmaSpace=75)
    k = max(8, int(12 - color_simp * 8))
    data = cv_img.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5), 5, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(cv_img.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges_inv = cv2.bitwise_not(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    result = cv2.bitwise_and(quantized, edges_inv)
    return to_pil(cv2.GaussianBlur(result, (3, 3), 0))

def apply_comic_book(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    k = max(6, int(10 - color_simp * 6))
    data = enhanced.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5), 6, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(enhanced.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=max(3, int(9 - line_w * 4)) | 1, C=2)
    kernel = np.ones((max(1, int(line_w * 2)), max(1, int(line_w * 2))), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return to_pil(cv2.bitwise_and(quantized, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))

def apply_charcoal(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur_size = int(5 + smooth * 10) | 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    sketch = cv2.divide(gray, blurred, scale=256)
    noise = np.random.normal(0, 25 * smooth, sketch.shape).astype(np.uint8)
    sketch = cv2.add(sketch, noise)
    edges = cv2.Canny(gray, 30, 100)
    sketch = cv2.addWeighted(sketch, 1, edges, 0.3, 0)
    return to_pil(cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR))

def apply_pop_art(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    k = max(4, int(8 - color_simp * 4))
    data = saturated.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5), 5, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(saturated.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 60)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=2)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return to_pil(cv2.bitwise_and(quantized, cv2.bitwise_not(edges_3ch)))

def apply_neon_glow(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=int(1 + line_w * 3))
    glow = np.zeros_like(cv_img)
    for i in range(3):
        glow[:, :, i] = cv2.bitwise_and(cv_img[:, :, i], edges)
    glow = cv2.GaussianBlur(glow, (15, 15), 0)
    result = cv2.addWeighted(cv_img, 0.3, glow, 0.7, 0)
    return to_pil(result)

def apply_vintage(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    sepia = np.array(to_pil(cv_img))
    sepia = cv2.transform(sepia, np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]))
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    grain = np.random.normal(0, 15, sepia.shape).astype(np.uint8)
    sepia = cv2.add(sepia, grain)
    h, w = sepia.shape[:2]
    kernel_x = cv2.getGaussianKernel(w, w / 3)
    kernel_y = cv2.getGaussianKernel(h, h / 3)
    mask = (kernel_y * kernel_x.T)
    mask = np.stack([mask / mask.max()] * 3, axis=2)
    return to_pil(cv2.cvtColor((sepia * mask).astype(np.uint8), cv2.COLOR_RGB2BGR))

def apply_mosaic(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    pixel_size = max(4, int(20 - smooth * 18))
    h, w = cv_img.shape[:2]
    small = cv2.resize(cv_img, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return to_pil(cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST))

def apply_thermal(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cmap = cv2.COLORMAP_HOT if color_simp > 0.5 else cv2.COLORMAP_JET
    return to_pil(cv2.applyColorMap(gray, cmap))

def apply_pastel(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    smooth_img = cv2.bilateralFilter(cv_img, d=9, sigmaColor=100, sigmaSpace=100)
    hsv = cv2.cvtColor(smooth_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.7, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    texture = np.random.normal(0, 5, result.shape).astype(np.uint8)
    return to_pil(cv2.addWeighted(result, 0.95, texture, 0.05, 0))

def apply_stained_glass(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    filtered = cv2.pyrMeanShiftFiltering(cv_img, sp=20, sr=30)
    k = max(6, int(12 - color_simp * 8))
    data = filtered.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5), 5, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(filtered.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=int(2 + line_w * 3))
    edges_3ch = cv2.bitwise_not(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    return to_pil(cv2.bitwise_and(quantized, edges_3ch))

def apply_ink_wash(img, line_w, smooth, detail, color_simp):
    # [Your existing code]
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur_size = int(15 + smooth * 20) | 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    wash = cv2.divide(gray, blurred, scale=200)
    wash = cv2.equalizeHist(wash)
    wash = cv2.addWeighted(wash, 0.8, np.zeros_like(wash), 0, 20)
    return to_pil(cv2.cvtColor(wash, cv2.COLOR_GRAY2BGR))

# ── Registry ──────────────────────────────────────────────────────────────────

PRESET_FILTERS = {
    "🎨 Ghibli Soft": apply_ghibli_soft,
    "⚡ Cell Shade": apply_cell_shade,
    "🖋️ B&W Ink": apply_bw_ink,
    "📐 Vector Flat": apply_vector_flat,
    "✏️ Pencil Sketch": apply_pencil_sketch,
    "💧 Watercolor": apply_watercolor,
    "📖 Comic Book": apply_comic_book,
    "🔥 Charcoal": apply_charcoal,
    "🎭 Pop Art": apply_pop_art,
    "💡 Neon Glow": apply_neon_glow,
    "📷 Vintage": apply_vintage,
    "🧩 Mosaic": apply_mosaic,
    "🌡️ Thermal": apply_thermal,
    "🌸 Pastel": apply_pastel,
    "🌈 Stained Glass": apply_stained_glass,
    "🀄 Ink Wash": apply_ink_wash,
}

PRESET_COLORS = {
    "🎨 Ghibli Soft": "#FF9A9E", "⚡ Cell Shade": "#a1c4fd", "🖋️ B&W Ink": "#cfd9df",
    "📐 Vector Flat": "#43e97b", "✏️ Pencil Sketch": "#bdc3c7", "💧 Watercolor": "#89f7fe",
    "📖 Comic Book": "#ff6b6b", "🔥 Charcoal": "#2c3e50", "🎭 Pop Art": "#f093fb",
    "💡 Neon Glow": "#00f260", "📷 Vintage": "#c4a484", "🧩 Mosaic": "#ffaaa5",
    "🌡️ Thermal": "#ff4b1f", "🌸 Pastel": "#fbc2eb", "🌈 Stained Glass": "#ff9a9e",
    "🀄 Ink Wash": "#4568dc",
}

FILTER_CATEGORIES = {
    "🖼️ Classic": ["🎨 Ghibli Soft", "⚡ Cell Shade", "🖋️ B&W Ink", "📐 Vector Flat"],
    "✏️ Sketch": ["✏️ Pencil Sketch", "🔥 Charcoal", "🀄 Ink Wash"],
    "🎨 Artistic": ["💧 Watercolor", "🌸 Pastel", "🎭 Pop Art"],
    "🎯 Effects": ["📖 Comic Book", "💡 Neon Glow", "🌡️ Thermal", "🧩 Mosaic", "📷 Vintage", "🌈 Stained Glass"],
}

# ── Profile Section Component ─────────────────────────────────────────────────

def show_user_info():
    """Display user info in a compact format next to upload"""
    user = st.session_state.get('current_user', {})
    username = user.get('name', user.get('username', 'User'))
    avatar_initial = username[0].upper() if username else 'U'
    
    user_info_html = f"""
    <div style="display: flex; align-items: center; gap: 0.5rem;">
        <div style="
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            font-weight: 600;
            color: white;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        ">
            {avatar_initial}
        </div>
        <span style="color: var(--text-secondary); font-size: 0.9rem; font-weight: 500;">
            {username}
        </span>
    </div>
    """
    return user_info_html

def show_settings_menu():
    """Display settings menu in a popover"""
    with st.popover("⚙️", use_container_width=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 👤 Account")
            if st.button("📋 Profile", use_container_width=True):
                st.session_state.show_profile_settings = True
            
            if st.button("🔒 Privacy", use_container_width=True):
                st.session_state.show_privacy = True
        
        with col2:
            st.markdown("##### ⚙️ Preferences")
            if st.button("🎨 Theme", use_container_width=True):
                st.session_state.show_theme = True
            
            if st.button("💾 Export", use_container_width=True):
                st.session_state.show_export_settings = True
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            st.session_state.authenticated = False
            st.session_state.jwt_token = None
            st.session_state.current_user = None
            st.session_state.uploaded_image = None
            st.rerun()

def show_profile_modal():
    """Display profile settings modal"""
    if st.session_state.get('show_profile_settings', False):
        with st.container():
            st.markdown("""
            <div style="
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: var(--bg-secondary);
                border-radius: 24px;
                padding: 2rem;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow-lg);
                z-index: 1000;
                width: 400px;
                max-width: 90%;
            ">
            """, unsafe_allow_html=True)
            
            st.markdown("#### ✏️ Edit Profile")
            
            user = st.session_state.get('current_user', {})
            
            with st.form("profile_form"):
                name = st.text_input("Name", value=user.get('name', ''))
                email = st.text_input("Email", value=user.get('email', ''), disabled=True)
                bio = st.text_area("Bio", placeholder="Tell us about yourself...")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("💾 Save"):
                        if 'current_user' in st.session_state:
                            st.session_state.current_user['name'] = name
                            st.session_state.current_user['bio'] = bio
                        st.success("Profile updated!")
                        st.session_state.show_profile_settings = False
                        st.rerun()
                
                with col2:
                    if st.form_submit_button("❌ Cancel"):
                        st.session_state.show_profile_settings = False
                        st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

def show_theme_settings():
    """Display theme settings modal"""
    if st.session_state.get('show_theme', False):
        with st.container():
            st.markdown("""
            <div style="
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: var(--bg-secondary);
                border-radius: 24px;
                padding: 2rem;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow-lg);
                z-index: 1000;
                width: 400px;
                max-width: 90%;
            ">
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🎨 Theme Settings")
            
            theme = st.selectbox("Theme", ["Dark (Default)", "Light", "System"], index=0)
            accent_color = st.color_picker("Accent Color", "#3B82F6")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Apply", use_container_width=True):
                    st.success("Theme updated!")
                    st.session_state.show_theme = False
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_theme = False
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

def show_privacy_settings():
    """Display privacy settings modal"""
    if st.session_state.get('show_privacy', False):
        with st.container():
            st.markdown("""
            <div style="
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: var(--bg-secondary);
                border-radius: 24px;
                padding: 2rem;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow-lg);
                z-index: 1000;
                width: 400px;
                max-width: 90%;
            ">
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🔒 Privacy Settings")
            
            st.checkbox("Keep my creations private", value=True)
            st.checkbox("Allow others to see my public gallery", value=False)
            st.checkbox("Receive email notifications", value=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save", use_container_width=True):
                    st.success("Privacy settings saved!")
                    st.session_state.show_privacy = False
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_privacy = False
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

def show_export_settings():
    """Display export settings modal"""
    if st.session_state.get('show_export_settings', False):
        with st.container():
            st.markdown("""
            <div style="
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: var(--bg-secondary);
                border-radius: 24px;
                padding: 2rem;
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow-lg);
                z-index: 1000;
                width: 400px;
                max-width: 90%;
            ">
            """, unsafe_allow_html=True)
            
            st.markdown("#### 💾 Export Settings")
            
            format = st.selectbox("Default Format", ["PNG", "JPG", "WEBP"], index=0)
            quality = st.slider("Quality", 0, 100, 95)
            resolution = st.selectbox("Resolution", ["Original", "2K", "4K", "8K"], index=0)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save", use_container_width=True):
                    st.success("Export settings saved!")
                    st.session_state.show_export_settings = False
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_export_settings = False
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

# ── Main App ──────────────────────────────────────────────────────────────────

def show_editor():
    st.set_page_config(page_title="TOONIFY", layout="wide", page_icon="🎨")

    if "active_preset" not in st.session_state:
        st.session_state.active_preset = list(PRESET_FILTERS.keys())[0]
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Split"
    if "show_profile_settings" not in st.session_state:
        st.session_state.show_profile_settings = False
    if "show_privacy" not in st.session_state:
        st.session_state.show_privacy = False
    if "show_theme" not in st.session_state:
        st.session_state.show_theme = False
    if "show_export_settings" not in st.session_state:
        st.session_state.show_export_settings = False

    # Show modals if active
    show_profile_modal()
    show_theme_settings()
    show_privacy_settings()
    show_export_settings()

    # Fixed Top Bar with CSS
    st.markdown("""
    <style>
    .fixed-top-bar {
        position: sticky;
        top: 0;
        z-index: 999;
        background: var(--bg-primary);
        padding: 1rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    .scrollable-sidebar {
        max-height: calc(100vh - 200px);
        overflow-y: auto;
        padding-right: 10px;
    }
    .scrollable-sidebar::-webkit-scrollbar {
        width: 6px;
    }
    .scrollable-sidebar::-webkit-scrollbar-track {
        background: var(--bg-tertiary);
        border-radius: 10px;
    }
    .scrollable-sidebar::-webkit-scrollbar-thumb {
        background: var(--accent-primary);
        border-radius: 10px;
    }
    .scrollable-sidebar::-webkit-scrollbar-thumb:hover {
        background: var(--accent-secondary);
    }
    </style>
    """, unsafe_allow_html=True)

    # Top bar with logo, view mode, user info, settings, and upload
    with st.container():
        st.markdown('<div class="fixed-top-bar">', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns([1.5, 2, 1, 0.5, 2])
        
        with col1:
            st.markdown("<span class='logo' style='font-size:1.8rem;font-weight:800;'>🎨 TOONIFY</span>", unsafe_allow_html=True)
        
        with col2:
            st.session_state.view_mode = st.radio(
                "", 
                ["Split", "Single"], 
                horizontal=True, 
                label_visibility="collapsed"
            )
        
        with col3:
            # User info
            user_info = show_user_info()
            st.markdown(user_info, unsafe_allow_html=True)
        
        with col4:
            # Settings menu as popover
            show_settings_menu()
        
        with col5:
            # File uploader
            uploaded = st.file_uploader(
                "", 
                type=["jpg", "jpeg", "png", "webp"], 
                label_visibility="collapsed",
                key="file_uploader"
            )
            if uploaded:
                st.session_state.uploaded_image = Image.open(uploaded)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content area
    canvas_col, sidebar_col = st.columns([2.6, 1], gap="large")

    with sidebar_col:
        st.markdown('<div class="scrollable-sidebar">', unsafe_allow_html=True)
        st.markdown("#### 🎨 Style Gallery")
        for cat, presets in FILTER_CATEGORIES.items():
            st.markdown(f"**{cat}**")
            cols = st.columns(2)
            for i, preset in enumerate(presets):
                with cols[i % 2]:
                    active = st.session_state.active_preset == preset
                    if st.button(preset, key=f"btn_{preset}", use_container_width=True,
                                 type="primary" if active else "secondary"):
                        st.session_state.active_preset = preset
                        st.rerun()

        st.markdown("---")
        st.markdown("#### ⚙️ Fine-Tuning")
        line_w = st.slider("🖊 Line Weight", 0.0, 1.0, 0.4, 0.05)
        color_simp = st.slider("🎨 Color Simplification", 0.0, 1.0, 0.75, 0.05)
        detail = st.slider("🔍 Detail Preservation", 0.0, 1.0, 0.25, 0.05)
        smooth = st.slider("✨ Smoothness", 0.0, 1.0, 0.5, 0.05)

        if "uploaded_image" in st.session_state and st.session_state.uploaded_image:
            styled_export = PRESET_FILTERS[st.session_state.active_preset](
                st.session_state.uploaded_image, line_w, smooth, detail, color_simp
            )
            buf = io.BytesIO()
            styled_export.save(buf, format="PNG")
            st.download_button("⬇ Export Image", buf.getvalue(), file_name="toonify_export.png", mime="image/png")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with canvas_col:
        if "uploaded_image" not in st.session_state or not st.session_state.uploaded_image:
            st.markdown(
                "<div class='upload-area' style='height:500px;border:2px dashed var(--border-color);"
                "border-radius:20px;display:flex;align-items:center;justify-content:center;"
                "font-size:1.2rem;color:var(--text-secondary);background:var(--bg-secondary);'>"
                "🎨 Upload a photo to get started</div>", unsafe_allow_html=True
            )
        else:
            img = st.session_state.uploaded_image
            styled = PRESET_FILTERS[st.session_state.active_preset](img, line_w, smooth, detail, color_simp)
            w, h = img.size
            
            # Image info bar
            st.markdown(f"""
            <div class='image-info'>
                <span class='image-dimensions'>📐 {w}×{h}px</span>
                <span class='active-filter'>🎨 {st.session_state.active_preset}</span>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.view_mode == "Split":
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**ORIGINAL**")
                    st.image(img, use_container_width=True)
                with c2:
                    st.markdown(f"**{st.session_state.active_preset}**")
                    st.image(styled, use_container_width=True)
            else:
                st.image(styled, use_container_width=True)

if __name__ == "__main__":
    show_editor()