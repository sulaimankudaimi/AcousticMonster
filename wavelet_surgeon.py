import pywt
import numpy as np
import matplotlib.pyplot as plt

def apply_wavelet_denoising(signal, wavelet='db4', level=2):
    """
    استخدام تفكيك المويجات (Wavelet Decomposition) 
    لعزل الضجيج عالي التردد دون التأثير على شكل قمة الصدى.
    """
    coeffs = pywt.wavedec(signal, wavelet, mode='per')
    # حساب حد العتبة (Threshold) لقص الضجيج
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # تطبيق العتبة الناعمة (Soft Thresholding)
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    
    return pywt.waverec(coeffs, wavelet, mode='per')

def extract_echo_features(signal, sampling_rate=5000):
    """
    تحويل الإشارة إلى Scalogram (خريطة زمنية-ترددية) 
    لتحديد موقع العائق بدقة جراحية.
    """
    widths = np.arange(1, 64)
    cfs, frequencies = pywt.cwt(signal, widths, 'mexh') # Mexican Hat wavelet
    return cfs

if __name__ == "__main__":
    # محاكاة إشارة صدى معقدة مع ضجيج
    t = np.linspace(0, 0.1, 500)
    # إشارة الصدى الحقيقية (ضعيفة ومغمورة في الضجيج)
    clean_signal = np.zeros_like(t)
    clean_signal[200:220] = 0.5 * np.sin(2 * np.pi * 100 * t[200:220])
    
    noisy_signal = clean_signal + np.random.normal(0, 0.2, 500)
    
    # إجراء "الجراحة"
    denoised = apply_wavelet_denoising(noisy_signal)
    scalogram = extract_echo_features(denoised)
    
    print("✅ Wavelet Surgery Complete. Signal Purified.")
    
    # عرض النتائج (رؤية الوحش الجديدة)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(noisy_signal, alpha=0.5, label="Noisy Signal")
    plt.plot(denoised, 'r', label="Denoised (Wavelet)")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.imshow(np.abs(scalogram), extent=[0, 100, 1, 64], cmap='jet', aspect='auto')
    plt.title("The Monster's Precision Vision (Scalogram)")
    plt.tight_layout()
    plt.show()