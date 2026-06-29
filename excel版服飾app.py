import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from sklearn.cluster import KMeans
from ultralytics import YOLO
import pandas as pd  
import io          
# === 1. 網頁基本設定 ===
st.set_page_config(page_title="智慧零售客群洞察", layout="wide")
st.title("智慧零售客群洞察與決策")

with st.sidebar:
    st.header(" 系統設定")
    api_key = st.text_input("請輸入 Gemini API Key", type="password")
    st.info("請輸入 API 金鑰以啟動大語言模型決策引擎。(此範例主要展示專家矩陣)")

uploaded_file = st.file_uploader("請上傳門店客流圖片 (JPG/PNG)", type=["jpg", "jpeg", "png"])

# === 2. 專家決策矩陣 ===
EXPERT_DECISION_MATRIX = {
    "body": {
        "沙漏型": {"剪裁": "合身剪裁、高腰收腰、X字廓形", "說明": "完美展現腰部曲線，避免過度寬鬆掩蓋身形優勢。"},
        "倒三角型": {"剪裁": "A字裙、闊腿褲、下身增加層次感、落肩設計", "說明": "弱化上肩視覺重量，平衡上下身比例。"},
        "梨型": {"剪裁": "上緊下鬆、高腰A字線條、直筒褲、有結構的上衣", "說明": "強調纖細上半身，優雅遮飾臀腿線條。"},
        "蘋果型": {"剪裁": "V領設計、高腰線剪裁、H字直筒線條、垂墜感面料", "說明": "延伸頸部線條，轉移中段視覺焦點。"},
        "矩形": {"剪裁": "腰帶收腰、層次感荷葉邊、Oversize搭配內緊外鬆", "說明": "人為製造曲線，增加整體視覺層次。"},
        "無法判斷": {"剪裁": "基礎標準直筒剪裁", "說明": "數據不足，建議採用最安全的經典版型。"}
    },
    "skin": {
        "暖色調": {"顏色": "杏色、奶油白、鵝黃、卡其色、珊瑚粉、磚紅、焦糖棕、香檳金", "說明": "襯托肌膚溫暖氣色，避免呈現暗沉。"},
        "冷色調": {"顏色": "冰藍、深海藍、薰衣草紫、冷灰色、純銀色、寶石紅、裸粉色", "說明": "突顯肌膚白皙清透感，避免顯得發黃。"},
        "中性色調": {"顏色": "經典純黑、高級灰、軍綠色、象牙白、大地色系", "說明": "百搭色調，任何服飾風格皆能輕鬆駕馭。"},
        "無法分析": {"顏色": "黑、白、灰黑基礎色系", "說明": "光線或取樣不足，建議採用安全中性色。"}
    }
}

# --- 膚色過濾與分類 ---
def is_skin(rgb):
    r, g, b = rgb
    return (r > 90) and (g > 40) and (b > 20) and ((max(rgb) - min(rgb)) > 15) and (abs(r - g) > 15) and (r > g) and (r > b)

def classify_skin_tone(rgb):
    r, g, b = rgb
    if r > g and r > b and r - b > 30: return "暖色調"
    elif b > r and b > g: return "冷色調"
    else: return "中性色調"

# === 3. 局部特徵提取與全局座標映射引擎 ===
def process_single_person(img_global, x1, y1, x2, y2):
    person_crop = img_global[y1:y2, x1:x2]
    if person_crop.size == 0: return "無法判斷", "無法分析"
    crop_h, crop_w, _ = person_crop.shape
    
    mp_face_detection = mp.solutions.face_detection
    skin_label = "無法分析"
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results_face = face_detection.process(person_crop)
        face_imgs = []
        if results_face.detections:
            for detection in results_face.detections:
                bbox = detection.location_data.relative_bounding_box
                fx, fy, fw, fh = int(bbox.xmin * crop_w), int(bbox.ymin * crop_h), int(bbox.width * crop_w), int(bbox.height * crop_h)
                regions = [
                    ((fx + int(fw * 0.2), fy + int(fh * 0.35)), (fx + int(fw * 0.4), fy + int(fh * 0.5))),
                    ((fx + int(fw * 0.6), fy + int(fh * 0.35)), (fx + int(fw * 0.8), fy + int(fh * 0.5))),
                    ((fx + int(fw * 0.35), fy + int(fh * 0.75)), (fx + int(fw * 0.65), fy + int(fh * 0.9)))
                ]
                for (rx1, ry1), (rx2, ry2) in regions:
                    ry1, ry2, rx1, rx2 = max(0, ry1), min(crop_h, ry2), max(0, rx1), min(crop_w, rx2)
                    region = person_crop[ry1:ry2, rx1:rx2]
                    if region.size > 0:
                        face_imgs.append(region)
                        cv2.rectangle(img_global, (x1 + rx1, y1 + ry1), (x1 + rx2, y1 + ry2), (255, 0, 0), 1)
        
        dominant_colors = []
        for region in face_imgs:
            pixels = region.reshape(-1, 3)
            skin_pixels = np.array([px for px in pixels if is_skin(px)])
            if len(skin_pixels) < 10: skin_pixels = pixels
            try:
                kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(skin_pixels)
                dominant_colors.extend(kmeans.cluster_centers_.astype(int))
            except: pass
        if dominant_colors: skin_label = classify_skin_tone(dominant_colors[0])

    mp_pose = mp.solutions.pose
    body_type = "無法判斷"
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results_pose = pose.process(person_crop)
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            def get_global_xy(idx): return (x1 + int(landmarks[idx].x * crop_w), y1 + int(landmarks[idx].y * crop_h))
                
            g_shoulder_left = get_global_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            g_shoulder_right = get_global_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            g_hip_l = get_global_xy(mp_pose.PoseLandmark.LEFT_HIP.value)
            g_hip_r = get_global_xy(mp_pose.PoseLandmark.RIGHT_HIP.value)
            
            # --- 升級版腰臀比例校正 ---
            g_hip_y = int((g_hip_l[1] + g_hip_r[1]) / 2)
            g_hip_width_raw = abs(g_hip_r[0] - g_hip_l[0])
            hip_compensation = int(g_hip_width_raw * 0.15) 
            g_hip_left = (min(g_hip_l[0], g_hip_r[0]) - hip_compensation, g_hip_y)
            g_hip_right = (max(g_hip_l[0], g_hip_r[0]) + hip_compensation, g_hip_y)
            
            g_mid_shoulder_y = int((g_shoulder_left[1] + g_shoulder_right[1]) / 2)
            g_waist_y = int(g_mid_shoulder_y * 0.6 + g_hip_y * 0.4) 
            
            s_w_raw = abs(g_shoulder_right[0] - g_shoulder_left[0])
            h_w_raw = g_hip_right[0] - g_hip_left[0]
            base_width = (s_w_raw + h_w_raw) / 2
            g_waist_width = int(base_width * 0.75) 
            g_mid_x = int((g_shoulder_left[0] + g_shoulder_right[0] + g_hip_l[0] + g_hip_r[0]) / 4)
            
            g_waist_left = (g_mid_x - int(g_waist_width / 2), g_waist_y)
            g_waist_right = (g_mid_x + int(g_waist_width / 2), g_waist_y)
            
            cv2.line(img_global, g_shoulder_left, g_shoulder_right, (0, 255, 0), 2)
            cv2.line(img_global, g_waist_left, g_waist_right, (0, 255, 0), 2)
            cv2.line(img_global, g_hip_left, g_hip_right, (0, 255, 0), 2)
            for pt in [g_shoulder_left, g_shoulder_right, g_waist_left, g_waist_right, g_hip_left, g_hip_right]:
                cv2.circle(img_global, pt, 4, (0, 255, 0), -1)
                
            s_w = np.linalg.norm(np.array(g_shoulder_right) - np.array(g_shoulder_left))
            w_w = np.linalg.norm(np.array(g_waist_right) - np.array(g_waist_left))
            h_w = np.linalg.norm(np.array(g_hip_right) - np.array(g_hip_left))
            
            ratio_sh = s_w / (h_w + 1e-6) 
            ratio_ws = w_w / (s_w + 1e-6) 
            ratio_wh = w_w / (h_w + 1e-6) 
            
            if 0.95 <= ratio_sh <= 1.5 and ratio_ws < 0.9 and 0.6 < ratio_wh < 0.9: body_type = "沙漏型"
            elif ratio_sh > 1.5: body_type = "倒三角型"
            elif ratio_sh < 0.95: body_type = "梨型"
            elif ratio_ws >= 0.9 and ratio_wh >= 0.9: body_type = "蘋果型"
            else: body_type = "矩形"
                
    return body_type, skin_label

# === 4. 主程式 Web Pipeline ===
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_rgb = np.array(image)
    img_display = img_rgb.copy()
    
    st.write("系統開始掃描...")
    model = YOLO('yolov8n.pt')
    results = model(img_rgb, verbose=False)
    
    store_body_stats = {"沙漏型": 0, "倒三角型": 0, "梨型": 0, "蘋果型": 0, "矩形": 0, "無法判斷": 0}
    store_skin_stats = {"暖色調": 0, "冷色調": 0, "中性色調": 0, "無法分析": 0}
    person_idx = 0
    
    # 新增：用來搜集資料匯出 Excel 的列表
    export_data_list = []
    
    with st.expander(" 查看個體特徵掃描明細"):
        for box in results[0].boxes:
            if int(box.cls[0]) == 0: 
                person_idx += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                body_type, skin_label = process_single_person(img_display, x1, y1, x2, y2)
                
                if body_type in store_body_stats: store_body_stats[body_type] += 1
                if skin_label in store_skin_stats: store_skin_stats[skin_label] += 1
                
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(img_display, f"P{person_idx}: {body_type}({skin_label})", (x1, y1 - 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                st.write(f"[P{person_idx}] 體型: {body_type} | 膚色: {skin_label}")
                
                # 新增：把每一筆資料存進列表
                export_data_list.append({
                    "顧客編號": f"P{person_idx}",
                    "偵測體型": body_type,
                    "偵測膚色": skin_label
                })

    st.image(img_display, caption=f"分析完成！共偵測到 {person_idx} 人", use_column_width=True)

    # === 5. 門店群體平均核心客群決策 ===
    st.markdown("---")
    st.header("門店群體平均決策報告")
    
    if person_idx > 0:
        valid_body_stats = {k: v for k, v in store_body_stats.items() if k != "無法判斷"}
        valid_skin_stats = {k: v for k, v in store_skin_stats.items() if k != "無法分析"}
        
        majority_body = max(valid_body_stats, key=valid_body_stats.get) if valid_body_stats else "無法判斷"
        majority_skin = max(valid_skin_stats, key=valid_skin_stats.get) if valid_skin_stats else "無法分析"
        
        st.subheader(f"核心客群畫像 ➔ 體型主導：【{majority_body}】 | 膚色主導：【{majority_skin}】")
        
        if majority_body != "無法判斷" and majority_skin != "無法分析":
            avg_body_rec = EXPERT_DECISION_MATRIX["body"][majority_body]
            avg_skin_rec = EXPERT_DECISION_MATRIX["skin"][majority_skin]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(" 主打剪裁策略")
                st.markdown(f"**建議剪裁：** {avg_body_rec['剪裁']}")
                st.markdown(f"**視覺優化目的：** {avg_body_rec['說明']}")
                
            with col2:
                st.info(" 季度備貨色彩行銷")
                st.markdown(f"**建議顏色：** {avg_skin_rec['顏色']}")
                st.markdown(f"**色彩行銷目的：** {avg_skin_rec['說明']}")
        else:
            st.warning("⚠️ 有效特徵數據不足，建議維持門店標準中性陳列。")
            
        # ==========================================================
        #將搜集的數據轉成 DataFrame 並提供 Excel 下載
        # ==========================================================
        st.markdown("---")
        st.subheader("資料匯出模組")
        
        # 1. 將清單轉換為 Pandas DataFrame (表格)
        df_export = pd.DataFrame(export_data_list)
        
        # 2. 在網頁上展示預覽表格
        st.write("門店客流分析明細 (預覽)：")
        st.dataframe(df_export, use_container_width=True)
        
        # 3. 建立記憶體緩衝區來產生 Excel 檔案
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='客流明細')
            
            # (進階) 你甚至可以把統計數據存到第二個工作表
            df_stats = pd.DataFrame([store_body_stats, store_skin_stats], index=["體型統計", "膚色統計"])
            df_stats.to_excel(writer, sheet_name='總計數據')
            
        excel_data = excel_buffer.getvalue()
        
        # 4. Streamlit 一鍵下載按鈕
        st.download_button(
            label="點擊下載完整 Excel 報表",
            data=excel_data,
            file_name="智慧門店_客流洞察報表.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
            
    else:
        st.error("畫面中未偵測到顧客。")