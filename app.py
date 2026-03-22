import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="Delivery Predictor", page_icon="🚚", layout="centered")

st.markdown("""
<style>
    .stApp { background:#0f0f0f; color:#fff; font-family:'Segoe UI',sans-serif; }
    [data-testid="stSidebar"] { display:none; }
    .header { text-align:center; padding:30px 0 10px; }
    .header h1 { color:#fff; font-size:2.2rem; margin:0; }
    .header p  { color:#888; font-size:1rem; margin:6px 0 0; }
    .card { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:14px; padding:24px; margin-bottom:20px; }
    .card-title { color:#f59e0b; font-size:.8rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:16px; }
    .result-delivered { background:linear-gradient(135deg,#16a34a,#15803d); border-radius:14px; padding:30px; text-align:center; }
    .result-cancelled { background:linear-gradient(135deg,#dc2626,#b91c1c); border-radius:14px; padding:30px; text-align:center; }
    .result-transit   { background:linear-gradient(135deg,#2563eb,#1d4ed8); border-radius:14px; padding:30px; text-align:center; }
    .result-delivered h2,.result-cancelled h2,.result-transit h2 { color:#fff; font-size:2rem; margin:0; }
    .result-delivered p,.result-cancelled p,.result-transit p { color:#fff; opacity:.9; margin:8px 0 0; font-size:1rem; }
    .conf { font-size:1.6rem; font-weight:700; margin-top:12px !important; }
    .warn { background:#292000; border:1px solid #f59e0b; border-radius:10px; padding:12px 16px; color:#f59e0b; font-size:.9rem; margin-top:16px; }
    div[data-testid="stForm"] { background:transparent; border:none; padding:0; }
    .stButton>button { background:#f59e0b; color:#000; font-weight:700; border:none; border-radius:30px; padding:14px 0; font-size:1.05rem; width:100%; cursor:pointer; margin-top:6px; }
    .stButton>button:hover { background:#d97706; }
    label { color:#aaa !important; font-size:.88rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Train Model with balancing ───────────────────────────────
@st.cache_resource(show_spinner="🔧 Preparing model...")
def train_model():
    df = pd.read_csv('delivery.csv')

    drop_cols = ['Order_ID','User_ID','Driver_ID','Order_Time','Delivery_Time',
                 'Restaurant_Lat','Restaurant_Lon','Customer_Lat','Customer_Lon',
                 'Driver_Lat','Driver_Lon']
    df = df.drop(columns=drop_cols, errors='ignore')

    # IQR clipping
    Q1 = df['Delivery_Distance_km'].quantile(0.25)
    Q3 = df['Delivery_Distance_km'].quantile(0.75)
    df['Delivery_Distance_km'] = np.clip(df['Delivery_Distance_km'], Q1-1.5*(Q3-Q1), Q3+1.5*(Q3-Q1))

    X = df.drop(columns=['Order_Status'])
    y = df['Order_Status']

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

    # ── Fix imbalance: oversample minority classes ───────────
    df_train = X.copy()
    df_train['Order_Status'] = y.values

    class_counts = df_train['Order_Status'].value_counts()
    max_count    = class_counts.max()

    balanced_dfs = []
    for cls in class_counts.index:
        cls_df = df_train[df_train['Order_Status'] == cls]
        if len(cls_df) < max_count:
            cls_df = resample(cls_df, replace=True, n_samples=max_count, random_state=42)
        balanced_dfs.append(cls_df)

    df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    X_bal = df_balanced.drop(columns=['Order_Status'])
    y_bal = df_balanced['Order_Status']

    X_train, _, y_train, _ = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

    enc    = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    scaler = StandardScaler()

    X_train_t = np.hstack([
        enc.fit_transform(X_train[cat_cols]),
        scaler.fit_transform(X_train[num_cols])
    ])

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight='balanced',   # extra safety on top of oversampling
        random_state=42
    )
    model.fit(X_train_t, y_train)

    return model, enc, scaler, cat_cols, num_cols

model, enc, scaler, CAT_COLS, NUM_COLS = train_model()

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🚚 Delivery Status Predictor</h1>
    <p>Fill in the order details below to predict the delivery outcome</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Form ─────────────────────────────────────────────────────
with st.form("predict_form"):

    st.markdown('<div class="card"><p class="card-title">📦 Order Information</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    restaurant_id = c1.number_input("Restaurant ID", 1, 1000, 100)
    quantity      = c2.number_input("Quantity", 1, 5, 2)
    total_price   = c3.number_input("Total Price (EGP)", 30.0, 1000.0, 270.0)

    c4, c5 = st.columns(2)
    item_name      = c4.selectbox("Item Name", ['Fried Chicken','Sandwich','Koshary','Sushi','Pizza','Burger','Pasta','Salad'])
    payment_method = c5.selectbox("Payment Method", ['Cash','Credit Card','Online'])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><p class="card-title">🗺️ Delivery Details</p>', unsafe_allow_html=True)
    c6, c7, c8 = st.columns(3)
    city              = c6.selectbox("City", ['Cairo','Giza','Alexandria','Mansoura','Tanta','Zagazig','Assiut'])
    delivery_distance = c7.number_input("Distance (km)", 0.1, 50.0, 2.0)
    delivery_duration = c8.number_input("Duration (mins)", 1, 120, 38)

    c9, c10 = st.columns(2)
    traffic_level  = c9.selectbox("Traffic Level", ['Low','Medium','High'])
    driver_vehicle = c10.selectbox("Driver Vehicle", ['Motorbike','Car','Bicycle'])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><p class="card-title">🧑‍✈️ Driver Info</p>', unsafe_allow_html=True)
    driver_availability = st.selectbox("Driver Availability", ['Online','Offline'])
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Predict Delivery Status")

# ── Prediction ───────────────────────────────────────────────
if submitted:
    input_df = pd.DataFrame([{
        'Restaurant_ID':             restaurant_id,
        'Item_Name':                 item_name,
        'Quantity':                  quantity,
        'Total_Price':               total_price,
        'Delivery_Duration_Minutes': delivery_duration,
        'City':                      city,
        'Payment_Method':            payment_method,
        'Driver_Vehicle':            driver_vehicle,
        'Delivery_Distance_km':      delivery_distance,
        'Traffic_Level':             traffic_level,
        'Driver_Availability':       driver_availability
    }])

    X_input = np.hstack([
        enc.transform(input_df[CAT_COLS]),
        scaler.transform(input_df[NUM_COLS].astype(float))
    ])

    pred    = model.predict(X_input)[0]
    proba   = model.predict_proba(X_input)[0]
    classes = model.classes_
    conf    = max(proba) * 100

    st.markdown("---")

    # Result banner
    if pred == 'Delivered':
        st.markdown(f"""
        <div class="result-delivered">
            <h2>✅ Delivered</h2>
            <p>Order has been successfully delivered to the customer.</p>
            <p class="conf">{conf:.1f}% confidence</p>
        </div>""", unsafe_allow_html=True)
    elif pred == 'Cancelled':
        st.markdown(f"""
        <div class="result-cancelled">
            <h2>❌ Cancelled</h2>
            <p>Order has been cancelled. Consider re-assigning a driver.</p>
            <p class="conf">{conf:.1f}% confidence</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-transit">
            <h2>🚚 In Transit</h2>
            <p>Order is currently on the way to the customer.</p>
            <p class="conf">{conf:.1f}% confidence</p>
        </div>""", unsafe_allow_html=True)

    # Low confidence warning
    if conf < 60:
        st.markdown(f'<div class="warn">⚠️ Low confidence ({conf:.1f}%). Prediction may not be reliable for this input combination.</div>', unsafe_allow_html=True)

    # Probability chart
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 2.5), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    color_map = {'Delivered':'#16a34a', 'Cancelled':'#dc2626', 'In Transit':'#2563eb'}
    bar_colors = [color_map.get(c, '#888') for c in classes]
    bars = ax.barh(classes, proba * 100, color=bar_colors, edgecolor='none', height=0.5)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Probability (%)", color='#aaa', fontsize=9)
    ax.tick_params(colors='white', labelsize=9)
    for s in ['top','right','bottom','left']: ax.spines[s].set_visible(False)
    for bar, val in zip(bars, proba * 100):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va='center', color='white', fontsize=10, fontweight='bold')
    ax.set_title("Prediction Probabilities", color='white', fontsize=11, pad=10)
    st.pyplot(fig, use_container_width=False)
    plt.close()