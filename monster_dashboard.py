import streamlit as st
import numpy as np
import asyncio
import threading
import time
import queue
import random

# --- [0. تأمين الاستيراد والربط البرمجي] ---
try:
    from streamlit.runtime.scriptrunner import add_script_run_context
except ImportError:
    try:
        from streamlit.runtime.script_run_context import add_script_run_context
    except ImportError:
        def add_script_run_context(thread): return thread

# محاولة استيراد درع الحماية وفك التشفير السيادي
try:
    from security.encryption_layer import DataShield
    shield = DataShield()
except ImportError:
    shield = None

# --- [1. المحرك التحليلي والتوأم الرقمي المصحح فيزيائياً] ---
# The single autonomous_manager lives in monster_core (STEP 5 consolidation).
from monster_core import autonomous_manager

def genetic_optimization_live(current_p):
    """محاكاة حساب نقطة التشغيل المثلى للضغط"""
    best_p = current_p + random.uniform(-1.5, 2.0)
    return max(0.0, best_p)

def generate_digital_twin_data():
    """توليد قراءات افتراضية مطابقة لفيزياء الحقل الحقيقي (وحدة الـ Bar) عند انقطاع السيرفر"""
    t = time.time()
    # ضغط تشغيلي حقيقي ومنطقي بالـ Bar يتأرجح حول الـ 25 بار
    base_pressure = 25.0 + 0.8 * np.sin(t * 0.5) + random.uniform(-0.1, 0.1)
    base_temp = 45.60 + 0.1 * np.sin(t * 0.1) + random.uniform(-0.01, 0.01)
    # مؤشر الفقد الفيزيائي (PINN Matrix Deviation Index)
    base_loss = 0.04 + 0.02 * np.cos(t * 0.2) + random.uniform(-0.005, 0.005)
    
    # توليد الإشارة الصوتية لتدفق النفط
    x = np.linspace(0, 500, 100)
    signal_wave = 2.0 * np.sin(x * 0.05 - t) + np.random.normal(0, 0.15, 100)
    signal_wave += base_loss * 15 * np.exp(-(x - 250)**2 / 2000)
    
    financial_exposure = base_loss * 350000.0
    
    return {
        "pressure": base_pressure,
        "temp": base_temp,
        "physics_loss": base_loss,
        "financial_risk": financial_exposure,
        "signal": signal_wave.tolist(),
        "status": "🔮 STANDALONE DIGITAL TWIN: ACTIVE (BALANCED OIL PHYSICS)"
    }

# --- [2. إعدادات الهوية والسيادة والصفحة] ---
st.set_page_config(page_title="Sovereign P.12 - Elite Command", layout="wide")

if 'data_queue' not in st.session_state:
    st.session_state.data_queue = queue.Queue()

if 'simulation_mode' not in st.session_state:
    st.session_state.simulation_mode = False  # جعل البث الحي من السيرفر هو الخيار الأساسي

# --- [3. هندسة الواجهة والتباين الدقيق والـ CSS الصناعي] ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #e0e0e0 !important; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #ffffff !important; }
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #00ff4133;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    [data-testid="stMetricValue"] {
        color: #00ff41 !important;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.4);
        font-weight: bold !important;
    }
    [data-testid="stMetricLabel"] {
        color: #00e5ff !important;
        font-weight: bold !important;
    }
    .decision-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 20px;
        border: 1px solid;
    }
    
    /* [إصلاح حاسم لمشكلة اختفاء نصوص الأزرار الشبحية] */
    .stButton > button, .stButton > button * {
        color: #0b0e14 !important; /* إجبار جميع النصوص الداخلية والفرعية على اللون الداكن المقروء */
        background-color: #ffffff !important;
        border: 2px solid #00ff41 !important;
        font-weight: bold !important;
        width: 100%;
        transition: 0.3s;
    }
    .stButton > button:hover, .stButton > button:hover * {
        background-color: #00ff41 !important;
        color: #000000 !important;
        box-shadow: 0 0 15px #00ff41;
    }
    </style>
    """, unsafe_allow_html=True)

# --- [4. تهيئة مخزن البيانات الموحد] ---
if 'live_data' not in st.session_state:
    st.session_state.live_data = {
        "pressure": 25.0, "temp": 45.0, "physics_loss": 0.02, "financial_risk": 0.0,
        "signal": [0]*100, "status": "🛰️ INITIALIZING SENSORS & BRIDGE..."
    }

# --- [5. محرك الجسر الخلفي والاتصال بالسيرفر الفوري عبر الـ WebSocket] ---
def start_bridge_thread(q):
    async def data_listener():
        uri = "ws://127.0.0.1:8000/ws/stream"
        import websockets
        import json
        import asyncio

        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    while True:
                        packet = await websocket.recv()
                        
                        if not packet or str(packet).strip() in ["", "b''", "''", '""']:
                            continue
                        
                        raw_packet = packet
                        
                        # محاولة فك التشفير السيادي بأمان
                        try:
                            if 'shield' in globals() and shield:
                                packet = shield.decrypt_data(packet)
                        except:
                            pass
                        
                        # معالجة حزمة البيانات المستلمة
                        try:
                            if isinstance(packet, bytes):
                                packet = packet.decode('utf-8')
                                
                            data_dict = json.loads(packet)
                            q.put(data_dict)
                        except json.JSONDecodeError as je:
                            sample = str(raw_packet)[:100]
                            print(f"❌ [DASHBOARD] حزمة تالفة هندسياً: {sample}... الخطأ: {je}")
                            await asyncio.sleep(0.1)
            except Exception as e:
                print(f"⚠️ [DASHBOARD BRIDGE] فقدت الإشارة السيرفرية: {e}. إعادة بناء الجسر بعد ثانيتين...")
                await asyncio.sleep(2)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(data_listener())

if 'bridge_active' not in st.session_state and not st.session_state.simulation_mode:
    thread = threading.Thread(target=start_bridge_thread, args=(st.session_state.data_queue,), daemon=True)
    add_script_run_context(thread)
    thread.start()
    st.session_state.bridge_active = True

# استهلاك حزم البيانات القادمة من السيرفر الحقيقي وتحديث واجهة المستخدم
if not st.session_state.simulation_mode:
    while not st.session_state.data_queue.empty():
        try:
            packet = st.session_state.data_queue.get_nowait()
            physics = packet.get("physics_vector", {})
            spatial = packet.get("spatial_data", {})
            financial = packet.get("financial_engine", {})
            
            st.session_state.live_data.update({
                "pressure": physics.get("pressure", 25.0), 
                "temp": physics.get("temperature", 45.0),
                "physics_loss": physics.get("physics_loss_index", 0.0),
                "financial_risk": financial.get("financial_risk_exposure_usd", 0.0),
                "signal": spatial.get("heatmap_energy", [0]*100),
                "status": "🟢 SOVEREIGN TELEMETRY: REAL-WORLD GPLA-12 ACTIVE STREAM"
            })
        except:
            break

if st.session_state.simulation_mode:
    st.session_state.live_data = generate_digital_twin_data()

# --- [6. معالجة الحسابات وعقلنة البيانات وضبط المخاطر احترازياً] ---
current_loss = float(st.session_state.live_data["physics_loss"])
current_risk_cost = float(st.session_state.live_data["financial_risk"])

decision_text, decision_color = autonomous_manager(current_loss, current_risk_cost)

# --- [7. بناء الهيكل المرئي والجمالي للوحة التحكم] ---
st.title("🚀 ACOUSTIC MONSTER | SOVEREIGN P.12")
st.caption(f"SYSTEM TELEMETRY STATUS: {st.session_state.live_data['status']} | INDUSTRIAL SYSTEM CONTROL")

# صندوق اتخاذ القرار التلقائي الذكي
st.markdown(f"""
    <div class="decision-box" style="background-color: {decision_color}22; border-color: {decision_color}; color: {decision_color} !important;">
        {decision_text}
    </div>
    """, unsafe_allow_html=True)

# عرض المؤشرات الفنية الأربعة الكبرى المصححة هندسياً
m1, m2, m3, m4 = st.columns(4)
with m1: 
    st.metric("PIPELINE PRESSURE", f"{st.session_state.live_data['pressure']:.2f} Bar")
with m2: 
    st.metric("PINN DEVIATION RESIDUAL", f"{current_loss:.4f} Index")
with m3: 
    st.metric("FINANCIAL RISK EXPOSURE", f"${current_risk_cost:,.2f}")
with m4: 
    st.metric("THERMAL MATRIX TEMP", f"{st.session_state.live_data['temp']:.2f} °C")

st.markdown("---")

col_main, col_ctrl = st.columns([2, 1])

with col_main:
    st.subheader("🔬 Signal Wavelet Surgeon (GPLA-12 Real-Time Stream Analysis)")
    st.line_chart(np.array(st.session_state.live_data["signal"]), color="#00ff41")
    
    st.markdown("### 🌐 Array Sensor Integrity Status")
    c1, c2, c3 = st.columns(3)
    c1.success("✅ Acoustic-Alpha Matrix (Online)")
    c2.success("✅ Wavelet Transient Node (Online)")
    c3.info("🔵 Digital Twin Synced Perfectly")

with col_ctrl:
    st.subheader("🕹️ Operational Directives")
    
    sim_toggle = st.toggle("Force Standalone Digital Twin Engine", value=st.session_state.simulation_mode)
    if sim_toggle != st.session_state.simulation_mode:
        st.session_state.simulation_mode = sim_toggle
        st.session_state.bridge_active = False 
        st.rerun()
        
    with st.expander("🛡️ Sovereign Advanced AI Parameters", expanded=True):
        st.toggle("Adaptive PINN Filter Boost", value=True)
        st.slider("Acoustic Feature Sensitivity", 0, 100, 85)
    
    if st.button("⚡ Trigger PINN Matrix Optimization"):
        opt_p = genetic_optimization_live(st.session_state.live_data["pressure"])
        st.info(f"Optimal Setpoint Calculated: {opt_p:.2f} Bar")

    if st.button("🛰️ Deploy Robotic Drone Recon"):
        st.toast("Drone Dispatched to Field Node P12. Scanning pipeline thermal signatures...")
        st.warning("Satellite Uplink Status: Stable | Encrypted GPS Locked")

    if st.button("🚀 Re-Sync Digital Core Bridge"):
        st.rerun()

# --- [8. محرك التحديث السريع والتدفق والإنعاش الحي للنفط] ---
time.sleep(0.15) 
st.rerun()