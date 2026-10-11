import numpy as np
import tensorflow as tf
from sovereign_brain import build_sovereign_model

print("🔥 Igniting the Chaos Forge (Robust Version)...")

# 1. بناء هيكل الوحش من جديد
model = build_sovereign_model()

# 2. تحميل الأوزان (الذكاء) فقط لتجنب أخطاء الصيغ
try:
    model.load_weights("sovereign_monster_v1.h5")
    print("✅ Intelligence Loaded Successfully.")
except:
    # في حال لم يقبل h5، سنستمر بالتدريب من الصفر على الضجيج
    print("⚠️ Notice: Starting fresh training for the hardened version.")

def add_industrial_noise(signal):
    # إضافة ضجيج أبيض (حساسات)
    white_noise = np.random.normal(0, 0.1, signal.shape)
    # إضافة ضجيج وردي محاكى (مضخات)
    pink_noise = np.interp(np.linspace(0, 1, len(signal)), np.linspace(0, 1, 10), np.random.normal(0, 0.2, 10))
    return signal + white_noise + pink_noise

# 3. توليد بيانات "قذرة" للتدريب (Chaos Data)
num_samples = 2000
time_steps = 500
t = np.linspace(0, 0.1, time_steps)

dirty_acoustic_data = []
for i in range(num_samples):
    clean_signal = np.sin(2 * np.pi * 10 * t) * np.exp(-5 * t)
    dirty_signal = add_industrial_noise(clean_signal)
    dirty_acoustic_data.append(dirty_signal)

dirty_acoustic_data = np.array(dirty_acoustic_data).reshape(num_samples, time_steps, 1)
env_data = np.random.uniform(low=[10, 20, 5], high=[55, 100, 500], size=(num_samples, 3))
targets = np.random.rand(num_samples, 3) 

# 4. إعادة تدريب "التحمل" (Hardening)
print("⚒️ Hardening the Monster against Industrial Noise...")
model.fit(
    [dirty_acoustic_data, env_data], 
    targets, 
    epochs=10, 
    batch_size=32
)

# 5. حفظ النسخة "المدرعة" بصيغة Keras الحديثة
model.save("hardened_monster_v2.keras")
print("\n[BATTLE READY] The Monster is now HARDENED and saved as 'hardened_monster_v2.keras'")