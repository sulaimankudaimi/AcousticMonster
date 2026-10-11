import numpy as np
import pywt
import matplotlib.pyplot as plt

# 1. إعداد البيانات (نبضة أصلية + صدى انسداد + ضجيج)
def generate_advanced_signal():
    t = np.linspace(0, 1, 1000)
    # النبضة المرسلة
    pulse = np.sin(2 * np.pi * 50 * t) * np.exp(-1000 * (t - 0.1)**2)
    # الصدى القادم من انسداد على بعد معين (تأخير زمني)
    delay = 0.45  # هذا هو الـ TDOA الذي سنبحث عنه
    echo = 0.4 * np.sin(2 * np.pi * 50 * (t - delay)) * np.exp(-1000 * (t - delay)**2)
    
    noise = np.random.normal(0, 0.05, 1000)
    return t, pulse + echo + noise, pulse

t, noisy_signal, reference_pulse = generate_advanced_signal()

# --- [الاداة 1: الموجات الصغيرة Wavelet Transform] ---
# استخدام Morlet wavelet للحصول على دقة زمنية ترددية
scales = np.arange(1, 128)
coefficients, frequencies = pywt.cwt(noisy_signal, scales, 'morl')

# --- [الاداة 2: الفلتر التكيفي البسيط LMS] ---
# وظيفتها: إزالة النبضة الأصلية لإظهار "الصدى" فقط
def simple_lms(signal, reference, mu=0.01):
    n = len(signal)
    filtered = np.zeros(n)
    error = np.zeros(n)
    w = 0.0 # معامل الفلتر
    for i in range(n):
        output = w * reference[i]
        error[i] = signal[i] - output
        w = w + mu * error[i] * reference[i]
    return error

echo_only = simple_lms(noisy_signal, reference_pulse)

# --- [الاداة 3: تحديد الموقع TDOA] ---
# البحث عن أعلى قمة في الإشارة المفلترة (بعد النبضة الأولى)
echo_zone = echo_only[200:] # نتجاهل أول 200 نقطة (مكان النبضة الأصلية)
peak_idx = np.argmax(np.abs(echo_zone)) + 200
tdoa_time = t[peak_idx] - 0.1 # الفرق الزمني بين النبضة والصدى
distance = (tdoa_time * 343) / 2 # المسافة (ذهاب وإياب)

# --- الرسم البياني الاحترافي ---
plt.figure(figsize=(12, 8))

# 1. الإشارة الأصلية
plt.subplot(3, 1, 1)
plt.plot(t, noisy_signal, label="Raw Signal with Noise")
plt.title("Surgical Analysis: Raw Input")
plt.legend()

# 2. نتيجة Wavelet (Scalogram)
plt.subplot(3, 1, 2)
plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap='jet', aspect='auto')
plt.title("Wavelet Transform (Morlet): Time-Frequency Localization")
plt.ylabel("Scales")

# 3. الصدى المستخلص (LMS) وتحديد الموقع
plt.subplot(3, 1, 3)
plt.plot(t, echo_only, color='red', label="Extracted Echo (LMS)")
plt.axvline(x=t[peak_idx], color='green', linestyle='--', label=f"Detected Echo at {distance:.2f}m")
plt.title("Adaptive Filtering & TDOA Localization")
plt.legend()

plt.tight_layout()
plt.show()

print(f"🎯 Precision Location: Blockage detected at {distance:.2f} meters.")