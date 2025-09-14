import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
import json
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ==================== إعداد الصفحة والتنسيق ====================
st.set_page_config(
    page_title="🚴‍♂️ منصة تنبؤ أسعار الدراجات الكهربائية",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تنسيقات CSS احترافية
st.markdown("""
<style>
    /* تنسيقات عامة */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* العنوان الرئيسي */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a202c;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        border-bottom: 4px solid #3498db;
        border-radius: 15px;
        background: linear-gradient(45deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }

    /* البطاقات */
    .prediction-card {
        background-color: #ffffff;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border-left: 6px solid #3498db;
        color: #2c3e50;
        transition: all 0.3s ease;
    }

    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }

    .info-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 18px;
        margin: 1.5rem 0;
        border-left: 6px solid #2ecc71;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }

    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    }

    /* الشريط الجانبي */
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 18px;
        margin: 1rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid rgba(52, 152, 219, 0.2);
    }

    /* حقول الإدخال */
    .stSelectbox>div>div>select, .stSlider>div>div>div>input {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
    }

    .stSelectbox>div>div>select:hover, .stSlider>div>div>div>input:hover {
        border-color: #3498db !important;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.2) !important;
    }

    /* الأزرار */
    .stButton>button {
        background: linear-gradient(45deg, #3498db, #2980b9) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }

    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15) !important;
        background: linear-gradient(45deg, #2980b9, #3498db) !important;
    }

    .stButton>button:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }

    /* زر التحميل */
    .stButton>button:after {
        content: "";
        display: block;
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.2), rgba(255,255,255,0));
        pointer-events: none;
    }

    /* النصوص في الشريط الجانبي */
    .sidebar .stSelectbox label, .sidebar .stSlider label {
        color: #2c3e50 !important;
        font-weight: 700 !important;
        font-size: 1.1rem;
        margin-bottom: 0.5rem !important;
    }

    /* رسوم بيانية */
    .chart-container {
        background-color: #ffffff;
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid rgba(52, 152, 219, 0.1);
    }

    /* مؤشرات */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 18px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border-left: 5px solid #3498db;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    }

    /* تأثيرات متحركة */
    .pulse {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }

    /* رسوم بيانية تفاعلية */
    .interactive-chart {
        background-color: #ffffff;
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
        border: 1px solid rgba(52, 152, 219, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== تحميل البيانات وتدريب النموذج ====================
@st.cache_resource
def load_and_train_model():
    """تحميل البيانات وتدريب النموذج"""
    try:
        # تحميل البيانات من ملف CSV
        df = pd.read_csv('ebike_large_dataset.csv')

        # معالجة البيانات
        le_brand = LabelEncoder()
        le_model = LabelEncoder()

        df['brand_encoded'] = le_brand.fit_transform(df['brand'])
        df['model_encoded'] = le_model.fit_transform(df['model'])

        # الميزات والهدف
        features = ['brand_encoded', 'model_encoded', 'battery_range', 'motor_power', 'weight', 'top_speed', 'has_gps', 'has_app', 'warranty']
        X = df[features]
        y = df['price']

        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # تدريب النماذج
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        gb_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

        # تقييم النماذج
        gb_score = gb_model.score(X_test, y_test)
        rf_score = rf_model.score(X_test, y_test)

        return gb_model, rf_model, le_brand, le_model, gb_score, rf_score, df
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل البيانات: {str(e)}")
        return None, None, None, None, None, None, None

# ==================== دالة التنبؤ ====================
def predict_price(brand, model_name, battery_range, motor_power, weight, top_speed, has_gps, has_app, warranty,
                  gb_model, rf_model, le_brand, le_model):
    """تنبؤ سعر الدراجة الكهربائية بناء على المدخلات"""
    try:
        # ترميز المدخلات
        brand_encoded = le_brand.transform([brand])[0]
        model_encoded = le_model.transform([model_name])[0]

        # تحضير البيانات للتنبؤ
        features = [[brand_encoded, model_encoded, battery_range, motor_power, weight, top_speed, has_gps, has_app, warranty]]

        # الحصول على تنبؤات من النماذج
        gb_prediction = gb_model.predict(features)[0]
        rf_prediction = rf_model.predict(features)[0]

        # حساب المتوسط
        avg_prediction = (gb_prediction + rf_prediction) / 2

        return gb_prediction, rf_prediction, avg_prediction
    except Exception as e:
        st.error(f"حدث خطأ أثناء التنبؤ: {str(e)}")
        return None, None, None

# ==================== دالة توليد بيانات وهمية ====================
def generate_dummy_data(num_samples=1000000):
    """توليد بيانات وهمية للدراجات الكهربائية"""
    # استخدام numpy arrays for better performance with large datasets
    brands = ['Tesla', 'VanMoof', 'Rad Power Bikes', 'Juiced Bikes', 'Aventon', 'Rungu', 'Juice', 'Super73', 'Charge', 'Sondors']
    models = {
        'Tesla': ['Cyberquad', 'Model S', 'Model X'],
        'VanMoof': ['S3', 'X3', 'V'],
        'Rad Power Bikes': ['RadRover', 'RadCity', 'RadWagon'],
        'Juiced Bikes': ['CrossCurrent', 'Odkin', 'Scrambler'],
        'Aventon': ['Sinch', 'Pace', 'Thousand'],
        'Rungu': ['Dualie', 'Trike', 'Barracuda'],
        'Juice': ['Bike', 'E-Bike', 'Mountain'],
        'Super73': ['Z1', 'R', 'S'],
        'Charge': ['City', 'Bike', 'Electric'],
        'Sondors': ['Fat', 'Original', 'Go']
    }

    # Pre-allocate arrays for better performance
    brand_data = np.empty(num_samples, dtype=object)
    model_data = np.empty(num_samples, dtype=object)
    battery_range_data = np.empty(num_samples, dtype=int)
    motor_power_data = np.empty(num_samples, dtype=int)
    weight_data = np.empty(num_samples, dtype=int)
    top_speed_data = np.empty(num_samples, dtype=int)
    has_gps_data = np.empty(num_samples, dtype=bool)
    has_app_data = np.empty(num_samples, dtype=bool)
    warranty_data = np.empty(num_samples, dtype=int)
    price_data = np.empty(num_samples, dtype=int)
    date_data = np.empty(num_samples, dtype=object)

    # Generate all data at once for better performance
    for i in range(num_samples):
        brand = np.random.choice(brands)
        model = np.random.choice(models[brand])

        # توليد مواصفات واقعية
        battery_range = np.random.randint(20, 150)  # مدى البطارية بين 20-150 كم
        motor_power = np.random.choice([250, 350, 500, 750, 1000, 1500, 2000])  # قوة المحرك

        # إضافة مواصفات إضافية
        weight = np.random.randint(15, 35)  # الوزن بالكيلوجرام
        top_speed = np.random.randint(20, 45)  # السرعة القصيرة بالكم/ساعة
        has_gps = np.random.choice([True, False])
        has_app = np.random.choice([True, False])
        warranty = np.random.choice([1, 2, 3, 5])  # فترة الضمان بالسنوات

        # حساب السعر بناء على عوامل واقعية
        base_price = 1500 + (battery_range * 15) + (motor_power * 1.5) - (weight * 50)

        # إضافة عوامل أخرى للسعر
        if brand in ['Tesla', 'VanMoof']:
            base_price *= 1.5  # ماركات فاخرة
        elif brand in ['Xiaomi', 'Segway']:
            base_price *= 0.7  # ماركات اقتصادية

        # إضافة عوامل أخرى
        if has_gps:
            base_price += 300
        if has_app:
            base_price += 200
        base_price += warranty * 150

        # إضافة بعض التقلبات العشوائية
        base_price *= np.random.uniform(0.9, 1.1)

        # التأكد من أن السعر ليس سالباً
        price = max(int(base_price), 500)

        # Store data in arrays
        brand_data[i] = brand
        model_data[i] = model
        battery_range_data[i] = battery_range
        motor_power_data[i] = motor_power
        weight_data[i] = weight
        top_speed_data[i] = top_speed
        has_gps_data[i] = has_gps
        has_app_data[i] = has_app
        warranty_data[i] = warranty
        price_data[i] = price
        date_data[i] = datetime.now() - timedelta(days=np.random.randint(0, 365))

    # Create DataFrame from arrays
    df = pd.DataFrame({
        'brand': brand_data,
        'model': model_data,
        'battery_range': battery_range_data,
        'motor_power': motor_power_data,
        'weight': weight_data,
        'top_speed': top_speed_data,
        'has_gps': has_gps_data,
        'has_app': has_app_data,
        'warranty': warranty_data,
        'price': price_data,
        'date': date_data
    })

    return df

# ==================== الواجهة الرئيسية ====================
def main():
    # تحميل البيانات وتدريب النموذج
    gb_model, rf_model, le_brand, le_model, gb_score, rf_score, df = load_and_train_model()

    if df is None:
        # إذا فشل تحميل البيانات، تولد بيانات وهمية
        df = generate_dummy_data()
        st.warning("تم توليد بيانات وهمية لأن ملف البيانات غير متوفر")

    # العنوان الرئيسي
    st.markdown('<h1 class="main-header">🚴‍♂️ منصة تنبؤ أسعار الدراجات الكهربائية</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666; margin-bottom: 2rem;">تنبؤات دقيقة لأسعار الدراجات</h3>', unsafe_allow_html=True)

    # قسم المقدمة
    st.markdown("""
    <div class="info-card">
        <h3 style="text-align: center; color: #2c3e50;">🚀 مرحباً بك في منصة التنبؤ المتقدمة</h3>
        <p style="text-align: center; color: #666;">استخدم الذكاء الاصطناعي للحصول على تنبؤات دقيقة لأسعار الدراجات الكهربائية</p>
    </div>
    """, unsafe_allow_html=True)

    # الشريط الجانبي
    with st.sidebar:
        st.markdown("### ⚙️ إعدادات التنبؤ")
        st.markdown("---")

        # اختيار الماركة
        brands = df['brand'].unique()
        brand = st.selectbox("🏷️ اختر الماركة:", sorted(brands))

        # اختيار الموديل بناء على الماركة
        model_options = df[df['brand'] == brand]['model'].unique()
        model_name = st.selectbox("🚲 اختر الموديل:", model_options, key="model_selector")

        battery_range = st.slider("🔋 مدى البطارية (km):", 20, 150, 50)
        motor_power = st.slider("⚡ قوة المحرك (W):", 250, 2000, 350)
        weight = st.slider("⚖️ الوزن (kg):", 15, 35, 20)
        top_speed = st.slider("🚀 السرعة القصوى (km/h):", 20, 45, 25)

        has_gps = st.checkbox("📡 نظام GPS", value=True)
        has_app = st.checkbox("📱 تطبيق الموبايل", value=True)
        warranty = st.slider("🛡️ فترة الضمان (سنوات):", 1, 5, 2)

        st.markdown("---")

        # زر التنبؤ
        predict_button = st.button("🚀 احسب السعر", use_container_width=True)

    # القسم الرئيسي
    if predict_button:
        # عرض مؤشرات التحميل
        with st.spinner("جاري حساب السعر..."):
            time.sleep(1.5)

            # التنبؤ بالسعر
            gb_prediction, rf_prediction, avg_prediction = predict_price(
                brand, model_name, battery_range, motor_power, weight,
                top_speed, has_gps, has_app, warranty, gb_model, rf_model, le_brand, le_model
            )

            if avg_prediction is not None:
                # عرض النتائج في بطاقة
                st.markdown("### 💰 نتائج التنبؤ")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("🌳 Gradient Boosting", f"${int(gb_prediction):,}")

                with col2:
                    st.metric("🌲 Random Forest", f"${int(rf_prediction):,}")

                with col3:
                    st.metric("📊 المتوسط", f"${int(avg_prediction):,}")

                # عرض التفاصيل
                st.markdown("### 📋 التفاصيل")
                col_spec1, col_spec2 = st.columns(2)

                with col_spec1:
                    st.metric("🏷️ الماركة", brand)
                    st.metric("🔋 مدى البطارية", f"{battery_range} km")
                    st.metric("⚖️ الوزن", f"{weight} kg")

                with col_spec2:
                    st.metric("🚲 الموديل", model_name)
                    st.metric("⚡ قوة المحرك", f"{motor_power} W")
                    st.metric("🚀 السرعة القصوى", f"{top_speed} km/h")
                    st.metric("🛡️ فترة الضمان", f"{warranty} سنوات")

                # تحليل التأثير
                st.markdown("### 📊 تحليل التأثير")
                col_eff1, col_eff2 = st.columns(2)

                with col_eff1:
                    st.metric("📈 تأثير البطارية", f"+${battery_range * 15:,.0f}")

                with col_eff2:
                    st.metric("⚡ تأثير المحرك", f"+${motor_power * 1.5:,.0f}")

                # رسوم بيانية تفاعلية
                st.markdown("### 📈 تحليل السوق")

                # رسم بياني لتوزيع الأسعار حسب الماركة
                fig_price_by_brand = px.box(df, x='brand', y='price', title='توزيع الأسعار حسب الماركة')
                fig_price_by_brand.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_price_by_brand, use_container_width=True)

                # رسم بياني لتوزيع الأسعار حسب مدى البطارية
                fig_price_by_battery = px.scatter(df, x='battery_range', y='price', color='brand',
                                                  title='العلاقة بين مدى البطارية والسعر')
                st.plotly_chart(fig_price_by_battery, use_container_width=True)

                # رسم بياني للمقارنة
                comparison_data = {
                    'النموذج': ['Gradient Boosting', 'Random Forest', 'المتوسط'],
                    'السعر': [gb_prediction, rf_prediction, avg_prediction]
                }
                comparison_df = pd.DataFrame(comparison_data)
                fig_comparison = px.bar(comparison_df, x='النموذج', y='السعر',
                                        title='مقارنة نتائج النماذج')
                st.plotly_chart(fig_comparison, use_container_width=True)

                # نصيحة
                st.markdown("""
                <div class="alert alert-info">
                    <p>💡 <strong>نصيحة:</strong> هذه التنبؤات مبنية على تحليل للبيانات المتاحة وقد تختلف في الواقع بناء على عوامل إضافية.</p>
                </div>
                """, unsafe_allow_html=True)

    # قسم إحصائيات التطبيق
    st.markdown("### 📊 إحصائيات التطبيق")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📈 دقة Gradient Boosting", f"{gb_score*100:.1f}%")

    with col2:
        st.metric("📉 دقة Random Forest", f"{rf_score*100:.1f}%")

    with col3:
        st.metric("🚴‍♂️ عدد الموديلات", len(df['model'].unique()))

    with col4:
        st.metric("🏷️ عدد الماركات", len(df['brand'].unique()))

# تشغيل التطبيق
if __name__ == "__main__":
    main()
