# web.py
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib, pickle
import shap
import matplotlib
import matplotlib.pyplot as plt

# 兼容 numpy 旧别名
if not hasattr(np, 'bool'):
    np.bool = bool

# ============== 字体/中文显示 ==================
def setup_chinese_font():
    """设置中文字体（优先系统字体，其次 ./fonts 目录）"""
    try:
        import matplotlib.font_manager as fm
        chinese_fonts = [
            'WenQuanYi Zen Hei','WenQuanYi Micro Hei','SimHei','Microsoft YaHei',
            'PingFang SC','Hiragino Sans GB','Noto Sans CJK SC','Source Han Sans SC'
        ]
        available = [f.name for f in fm.fontManager.ttflist]
        for f in chinese_fonts:
            if f in available:
                matplotlib.rcParams['font.sans-serif'] = [f, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                return f

        # 尝试加载 ./fonts 下自带字体
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        candidates = [
            'NotoSansSC-Regular.otf','NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf','SimHei.ttf','MicrosoftYaHei.ttf'
        ]
        if os.path.isdir(fonts_dir):
            import matplotlib.font_manager as fm
            for fname in candidates:
                fpath = os.path.join(fonts_dir, fname)
                if os.path.exists(fpath):
                    fm.fontManager.addfont(fpath)
                    fam = fm.FontProperties(fname=fpath).get_name()
                    matplotlib.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial']
                    matplotlib.rcParams['font.family'] = 'sans-serif'
                    return fam
    except Exception:
        pass

    # 兜底
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    return None

chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = matplotlib.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============== 页面配置 ==================
st.set_page_config(
    page_title="Gastrointestinal Diseases Patients Aged 60 - 85 Years' Depression Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# ============== 特征与中文名 ==================
feature_names_display = [
    'Gender',
    'Pain',
    'Retire',
    'Falldown',
    'Disability',
    'Self_perceived_health',
    'Life_satisfaction',
    'Eyesight',
    'ADL_score',
    'Sleep_time'
]
feature_names_cn = [
    'Gender','Pain','Retirement','Fall','Disability',
    'Self-rated health','Life satisfaction','Vision','ADL (0–6)','Sleep time (hours)'
]
feature_dict = dict(zip(feature_names_display, feature_names_cn))
feature_dict.update({'ADL_score': 'ADL (0–6)', 'Eyesight': 'Vision'})  # 别名映射

variable_descriptions = {
    'Gender': 'Sex: Male=1, Female=2',
    'Pain': 'Pain: Yes=1, No=0',
    'Retire': 'Retired: Yes=1, No=0',
    'Falldown': 'Fall: Yes=1, No=0',
    'Disability': 'Disability: Yes=1, No=0',
    'Self_perceived_health': 'Self-rated health: Poor=1, Fair=2, Good=3',
    'Life_satisfaction': 'Life satisfaction: Poor=1, Fair=2, Good=3',
    'Eyesight': 'Vision: Poor=1, Fair=2, Good=3',
    'ADL_score': 'Activities of Daily Living (0–6)',
    'Sleep_time': 'Sleep time (hours)'
}

# ============== 工具函数 ==================
def _clean_number(x):
    """把 '[3.3101046E-1]'、'3,210'、' 12. ' 等转成 float；失败返回 NaN"""
    if isinstance(x, str):
        s = x.strip().strip('[](){}').replace(',', '')
        try:
            return float(s)
        except Exception:
            return np.nan
    return x

@st.cache_resource
def load_model(model_path: str = './ann_model.pkl'):
    """Load ANN model (scikit-learn style), with light compatibility for older pickles."""
    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        # 尝试获取特征名（优先 sklearn 风格）
        model_feature_names = None
        try:
            if hasattr(model, 'feature_names_in_'):
                model_feature_names = list(model.feature_names_in_)
        except Exception:
            pass

        return model, model_feature_names
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def predict_proba_safe(model, X_df):
    """Use sklearn-style predict_proba for ANN; raise a clear error if unavailable."""
    try:
        return model.predict_proba(X_df)
    except Exception as e:
        raise RuntimeError(f"Model does not support predict_proba correctly: {e}")

# ============== 主逻辑 ==================
def main():
    # 侧边栏
    st.sidebar.title("Gastrointestinal Diseases Patients Aged 60 - 85 Years' Depression Risk Predictor")
    st.sidebar.image(
        "https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg",
        width=200
    )
    st.sidebar.markdown("""
    ### About
    This tool uses an Artificial Neural Network (ANN) to estimate depression risk
    in patients aged 60–85 years with gastrointestinal diseases.

    **Outputs:**
    - Probability of depression vs no depression
    - Risk stratification (Low/Moderate/High)
    - SHAP-based model explanation

    """)
    with st.sidebar.expander("Feature Description"):
        for f in feature_names_display:
            st.markdown(f"**{feature_dict.get(f, f)}**: {variable_descriptions.get(f, '')}")

    st.title("Gastrointestinal Diseases Patients Aged 60 - 85 Years' Depression Risk Predictor")
    st.markdown("### Please enter all features below and click Predict")

    # 加载模型
    try:
        model, model_feature_names = load_model('./ann_model.pkl')
        st.sidebar.success("Model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Model load failed: {e}")
        return

    # 输入区域
    st.header("Patient Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female", index=0)
        pain = st.selectbox("Pain", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        retire = st.selectbox("Retired", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        falldown = st.selectbox("Fall", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
    with c2:
        disability = st.selectbox("Disability", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        three_map = {1: 'Poor', 2: 'Fair', 3: 'Good'}
        self_health = st.selectbox("Self-rated health", options=[1, 2, 3], format_func=lambda x: three_map.get(x, str(x)), index=1)
        life_satis = st.selectbox("Life satisfaction", options=[1, 2, 3], format_func=lambda x: three_map.get(x, str(x)), index=1)
        eyesight = st.selectbox("Vision", options=[1, 2, 3], format_func=lambda x: three_map.get(x, str(x)), index=1)
    with c3:
        adl = st.number_input("ADL (0–6)", value=0, step=1, min_value=0, max_value=6)
        sleep_time = st.number_input("Sleep time (hours)", value=7.0, step=0.5)

    if st.button("Predict", type="primary"):
        # 组装输入
        user_inputs = {
            'Gender': gender,
            'Pain': pain,
            'Retire': retire,
            'Falldown': falldown,
            'Disability': disability,
            'Self_perceived_health': self_health,
            'Life_satisfaction': life_satis,
            'Eyesight': eyesight,
            'ADL_score': adl,
            'Sleep_time': sleep_time,
        }

        # 特征名对齐（别名→页面键）
        alias_to_user_key = {
            'gender': 'Gender','sex':'Gender',
            'pain':'Pain',
            'retire':'Retire','retired':'Retire',
            'fall':'Falldown','fall_down':'Falldown','falldown':'Falldown',
            'disability':'Disability',
            'self_perceived_health':'Self_perceived_health','self_rated_health':'Self_perceived_health',
            'life_satisfaction':'Life_satisfaction','life_sat':'Life_satisfaction',
            'eyesight':'Eyesight','vision':'Eyesight',
            'ADL':'ADL_score','adl':'ADL_score','ADLscore':'ADL_score',
            'sleep_time':'Sleep_time','sleep':'Sleep_time'
        }

        # 构造输入 DataFrame
        if model_feature_names:
            resolved_values, missing_features = [], []
            for c in model_feature_names:
                ui_key = alias_to_user_key.get(c, c)
                val = user_inputs.get(ui_key, None)
                if val is None:
                    missing_features.append(c)
                resolved_values.append(val)
            if missing_features:
                st.error(f"The following model features are missing or mismatched: {missing_features}")
                with st.expander("Debug: Model vs Input feature names"):
                    st.write("Model feature names:", model_feature_names)
                    st.write("Page input keys:", list(user_inputs.keys()))
                return
            input_df = pd.DataFrame([resolved_values], columns=model_feature_names)
        else:
            input_df = pd.DataFrame([[user_inputs[c] for c in feature_names_display]], columns=feature_names_display)

        # 清洗 & 转数值
        input_df = input_df.applymap(_clean_number)
        for c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors='coerce')
        if input_df.isnull().any().any():
            st.error("There are missing/unparseable inputs. Please check the format (numbers without brackets).")
            with st.expander("Debug: Current Input DataFrame"):
                st.write(input_df)
            return

        # ======== 预测 ========
        try:
            proba = predict_proba_safe(model, input_df)[0]
            if len(proba) == 2:
                pos_idx = 1
                try:
                    if hasattr(model, 'classes_') and 1 in list(getattr(model, 'classes_', [])):
                        import numpy as _np
                        pos_idx = int(_np.where(_np.array(model.classes_) == 1)[0][0])
                except Exception:
                    pos_idx = 1
                dep_prob = float(proba[pos_idx])
                no_dep_prob = float(1.0 - dep_prob)
            else:
                raise ValueError("Unexpected probability shape")

            # 展示结果
            st.header("Depression Risk Prediction Result")
            a, b = st.columns(2)
            with a:
                st.subheader("No Depression Probability")
                st.progress(no_dep_prob)
                st.write(f"{no_dep_prob:.2%}")
            with b:
                st.subheader("Depression Probability")
                st.progress(dep_prob)
                st.write(f"{dep_prob:.2%}")

            risk_level = "Low risk" if dep_prob < 0.3 else ("Moderate risk" if dep_prob < 0.7 else "High risk")
            risk_color = "green" if dep_prob < 0.3 else ("orange" if dep_prob < 0.7 else "red")
            st.markdown(f"### Depression Risk Assessment: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)

            # ======= SHAP 解释 =======
            st.write("---"); st.subheader("Model Explanation (SHAP)")
            try:
                # 优先通用入口
                try:
                    explainer = shap.Explainer(model)
                    sv = explainer(input_df)  # Explanation
                    shap_value = np.array(sv.values[0])
                    expected_value = sv.base_values[0] if np.ndim(sv.base_values) else sv.base_values
                except Exception:
                    # 回退 TreeExplainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_df)
                    if isinstance(shap_values, list):
                        shap_value = np.array(shap_values[1][0])
                        ev = explainer.expected_value
                        expected_value = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
                    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                        shap_value = shap_values[0, :, 1]
                        ev = explainer.expected_value
                        expected_value = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
                    else:
                        shap_value = np.array(shap_values[0])
                        expected_value = explainer.expected_value

                current_features = list(input_df.columns)

                # --- 瀑布图 ---
                st.subheader("SHAP Waterfall Plot")
                import matplotlib.font_manager as fm
                try:
                    c_fonts = [
                        'WenQuanYi Zen Hei','WenQuanYi Micro Hei','Noto Sans CJK SC',
                        'Source Han Sans SC','SimHei','Microsoft YaHei','PingFang SC','Hiragino Sans GB'
                    ]
                    avail = [f.name for f in fm.fontManager.ttflist]
                    for f in c_fonts:
                        if f in avail:
                            plt.rcParams['font.sans-serif'] = [f, 'DejaVu Sans']; break
                except Exception:
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans','Arial']
                plt.rcParams['axes.unicode_minus'] = False

                fig_waterfall = plt.figure(figsize=(12, 8))
                display_data = input_df.iloc[0].copy()
                # 映射离散变量为可读英文
                try:
                    if 'Gender' in display_data.index:
                        display_data['Gender'] = {1:'Male',2:'Female'}.get(int(display_data['Gender']), display_data['Gender'])
                    for b in ['Pain','Retire','Falldown','Disability']:
                        if b in display_data.index:
                            display_data[b] = {0:'No',1:'Yes'}.get(int(display_data[b]), display_data[b])
                    for t in ['Self_perceived_health','Life_satisfaction','Eyesight']:
                        if t in display_data.index:
                            display_data[t] = {1:'Poor',2:'Fair',3:'Good'}.get(int(display_data[t]), display_data[t])
                except Exception:
                    pass

                try:
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,
                            base_values=expected_value,
                            data=display_data.values,
                            feature_names=[feature_dict.get(f, f) for f in current_features]
                        ),
                        max_display=len(current_features),
                        show=False
                    )
                except Exception:
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,
                            base_values=expected_value,
                            data=display_data.values,
                            feature_names=current_features
                        ),
                        max_display=len(current_features),
                        show=False
                    )

                # 修正 Unicode 负号，强制字体
                for ax in fig_waterfall.get_axes():
                    for text in ax.texts:
                        s = text.get_text()
                        if '−' in s: text.set_text(s.replace('−','-'))
                        if chinese_font: text.set_fontfamily(chinese_font)
                    for label in ax.get_yticklabels() + ax.get_xticklabels():
                        t = label.get_text()
                        if '−' in t: label.set_text(t.replace('−','-'))
                        if chinese_font: label.set_fontfamily(chinese_font)
                    if chinese_font:
                        ax.set_xlabel(ax.get_xlabel(), fontfamily=chinese_font)
                        ax.set_ylabel(ax.get_ylabel(), fontfamily=chinese_font)
                        ax.set_title(ax.get_title(), fontfamily=chinese_font)

                plt.tight_layout()
                st.pyplot(fig_waterfall); plt.close(fig_waterfall)

                # --- 力图 ---
                st.subheader("SHAP Force Plot")
                try:
                    import streamlit.components.v1 as components
                    force_plot = shap.force_plot(
                        expected_value,
                        shap_value,
                        display_data,
                        feature_names=[feature_dict.get(f, f) for f in current_features]
                    )
                    shap_html = f"""
                    <head>{shap.getjs()}</head>
                    <body><div class="force-plot-container">{force_plot.html()}</div></body>
                    """
                    components.html(shap_html, height=400, scrolling=False)
                except Exception as e:
                    st.warning(f"Force plot generation failed: {e}")

            except Exception as e:
                st.error(f"Failed to generate SHAP explanation: {e}")
                import traceback; st.error(traceback.format_exc())

        except Exception as e:
            st.error(f"Prediction or visualization failed: {e}")
            import traceback; st.error(traceback.format_exc())

    st.write("---")
    st.caption("© Depression Risk Predictor (ANN)")

if __name__ == "__main__":
    main()
