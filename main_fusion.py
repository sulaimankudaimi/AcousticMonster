import os
import numpy as np
import asyncio
import traceback
import time

# [1] تهيئة البيئة: إسكات تنبيهات TensorFlow غير الضرورية لتركيز الانتباه على البيانات
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

try:
    from sovereign_brain import build_sovereign_model
    from bridge_client import SovereignBridgeClient
    from monster_detector import monster_decision_unit
    print("✅ [SYSTEM] All Fusion Subsystems Synchronized.")
except ImportError as e:
    print(f"❌ [CRITICAL ERROR] Failed to load core modules: {e}")
    exit(1)

# تأمين ثبات النتائج التجريبية
np.random.seed(42)
tf.random.set_seed(42)

print("🦾 [START] Initializing Acoustic Monster Fusion Core v4.0...")

# [2] بناء الدماغ السيادي (The Sovereign Brain)
model = build_sovereign_model()
weights_path = "models/sovereign_monster_bayesian.weights.h5"

if os.path.exists(weights_path):
    try:
        model.load_weights(weights_path)
        print("🛡️ [AI] Bayesian Weights Loaded Successfully.")
    except Exception as e:
        print(f"⚠️ [AI WARNING] Weights sync failed ({e}), operating in default mode.")
else:
    print("💡 [AI INFO] Initial learning mode active (No pre-trained weights).")

# [3] معالج الانصهار الحي (The Live Fusion Processor)
async def live_fusion_processor(packet):
    """
    الدالة المسؤولة عن تحويل "حزمة البيانات" إلى "قرار فيزيائي".
    يتم استدعاؤها تلقائياً بواسطة الجسر عند وصول كل نبضة.
    """
    try:
        # أ) تفكيك الحزمة الموحدة
        physics = packet.get("physics_vector", {})
        spatial = packet.get("spatial_data", {})
        header = packet.get("header", {})
        
        # ب) معالجة مدخلات البيئة (Temperature, Pressure, Velocity)
        # نستخدم float() لضمان التوافق مع مصفوفات TensorFlow
        env_input = np.array([[
            float(physics.get("temperature", 0)), 
            float(physics.get("pressure", 0)), 
            float(physics.get("velocity", 0))
        ]]).astype(np.float32)
        
        # ج) معالجة إشارة الخريطة الحرارية (Heatmap/Acoustic Signal)
        signal_raw = np.array(spatial.get("heatmap_energy", [])).astype(np.float32)
        
        # تدقيق حجم الإشارة (يجب أن يكون 500 عنصر للتحليل السمعي)
        if signal_raw.shape[0] != 500:
            # إعادة ضبط الحجم تلقائياً في حال وجود تفاوت بسيط في البث
            signal_raw = np.pad(signal_raw, (0, max(0, 500 - len(signal_raw))), 'constant')[:500]
            
        signal_input = signal_raw.reshape(1, 500, 1)
        
        # د) التمرير لوحدة القرار (The Decision Unit)
        # هنا يتم الربط مع العرض المرئي والتحليل في monster_detector
        monster_decision_unit(signal_input, env_input, model, packet)
        
    except Exception as e:
        print(f"❌ [FUSION ERROR] Data processing failed: {e}")
        # traceback.print_exc() # قم بإلغاء التعليق هنا إذا أردت رؤية تفاصيل الخطأ البرمجي

# [4] تشغيل وإدارة الجسر السيادي
async def start_fusion_engine():
    print("🛰️ [BRIDGE] Activating Sovereign P.12 Reception Bridge...")
    
    try:
        # إنشاء العميل وربطه بملف المفتاح
        bridge = SovereignBridgeClient(key_path=r"F:\AcousticMonster\models\sovereign_vault.key")
        
        # بدء الاستماع للبث وتمرير كل حزمة لـ 'live_fusion_processor'
        await bridge.start_bridge(live_fusion_processor)
        
    except Exception as e:
        print(f"❌ [CRASH] Structural failure in Fusion Engine: {e}")

if __name__ == "__main__":
    try:
        # تشغيل المحرك في حلقة أحداث Asyncio
        asyncio.run(start_fusion_engine())
    except KeyboardInterrupt:
        print("\n🛑 [SHUTDOWN] Sovereign System Terminated by User.")
    except Exception as e:
        print(f"💀 [FATAL] System collapsed: {e}")