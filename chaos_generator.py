import numpy as np
import matplotlib.pyplot as plt

def generate_chaos_scenario(num_samples=1000, noise_level=0.1, turbulence_factor=0.05):
    """
    توليد إشارة صوتية تحتوي على فوضى صناعية وضوضاء بيئية.
    """
    t = np.linspace(0, 1, num_samples)
    
    # 1. الإشارة الأساسية (الموجة النظيفة)
    # نفترض نبضة صوتية أساسية
    clean_signal = np.sin(2 * np.pi * 5 * t) * np.exp(-20 * (t - 0.2)**2)
    
    # 2. إضافة "الانسداد الكاذب" (Echo) 
    # صدى ناتج عن انسداد مفترض عند زمن 0.6
    echo = 0.3 * np.sin(2 * np.pi * 5 * (t - 0.4)) * np.exp(-20 * (t - 0.6)**2)
    clean_signal += echo

    # 3. حقن الفوضى (Chaos Injection):
    
    # أ. الضوضاء البيضاء (White Noise) - تمثل ضجيج الحساسات
    white_noise = np.random.normal(0, noise_level, num_samples)
    
    # ب. الضجيج الوردي/المستمر (Pink/Pump Noise) - تمثل ضجيج المضخات
    pump_noise = 0.05 * np.sin(2 * np.pi * 50 * t) # ضجيج عالي التردد من المحركات
    
    # ج. الاضطراب (Turbulence) - تغييرات عشوائية في السعة بنسبة 5-10%
    turbulence = 1 + np.random.uniform(-turbulence_factor, turbulence_factor, num_samples)
    
    # 4. تجميع "الإشارة القذرة"
    dirty_signal = (clean_signal * turbulence) + white_noise + pump_noise
    
    return t, clean_signal, dirty_signal

# --- تشغيل المولد لسيناريو واحد للمعاينة ---
t, clean, dirty = generate_chaos_scenario(noise_level=0.15, turbulence_factor=0.1)

# رسم المقارنة بين المثالي والواقعي
plt.figure(figsize=(15, 6))
plt.plot(t, dirty, color='red', alpha=0.5, label='Dirty Data (Industrial Chaos)')
plt.plot(t, clean, color='black', linewidth=2, label='Ideal Physics (The Lab)')
plt.title("Phase 2: The Chaos Forge - Ideal vs. Reality")
plt.xlabel("Time (seconds)")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("[SUCCESS] Chaos Forge is hot! Dirty data generated.")