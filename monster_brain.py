import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. دالة توليد البيانات بكميات كبيرة (Data Factory)
def generate_dataset(num_samples=5000):
    t = np.linspace(0, 1, 500)
    clean_data = []
    dirty_data = []
    
    for _ in range(num_samples):
        # توليد تردد ومكان عشوائي للنبضة لمحاكاة انسدادات مختلفة
        freq = np.random.uniform(3, 7)
        pos = np.random.uniform(0.1, 0.4)
        
        clean = np.sin(2 * np.pi * freq * t) * np.exp(-20 * (t - pos)**2)
        
        # إضافة ضوضاء واضطراب
        noise = np.random.normal(0, 0.15, 500)
        dirty = clean + noise + 0.05 * np.sin(2 * np.pi * 50 * t)
        
        clean_data.append(clean)
        dirty_data.append(dirty)
        
    return np.array(dirty_data), np.array(clean_data)

# 2. تحضير البيانات
print("Generating 5000 Industrial Scenarios...")
X_train, y_train = generate_dataset(5000)

# 3. بناء "العقل المرشح" (The Denoising Autoencoder)
model = models.Sequential([
    layers.Input(shape=(500,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'), # عنق الزجاجة (حيث يركز على الفيزياء)
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(500, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# 4. التدريب (تعليم الوحش كيف ينظف البيانات)
print("Training the Brain to see through Chaos...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# 5. اختبار العقل على عينة جديدة تماماً
test_dirty, test_clean = generate_dataset(1)
predicted_clean = model.predict(test_dirty)

# رسم النتيجة
plt.figure(figsize=(12, 5))
plt.plot(test_dirty[0], color='red', alpha=0.3, label='Input: Dirty Signal')
plt.plot(test_clean[0], color='black', label='Target: Pure Physics')
plt.plot(predicted_clean[0], color='blue', linewidth=2, label='Monster Output: Reconstructed Signal')
plt.title("Monster Brain: Successfully Extracting Physics from Chaos")
plt.legend()
plt.show()

# --- المرحلة الثالثة: نظام التشخيص الذكي ---
print("\nMoving to Phase 3: Diagnostic Radar Installation...")

# 1. دالة توليد بيانات مع "تسميات" (Labels) للانسداد
def generate_diagnostic_data(num_samples=2000):
    t = np.linspace(0, 1, 500)
    X = [] # الإشارة القذرة
    Y = [] # [هل يوجد انسداد؟ ، مكانه]
    
    for _ in range(num_samples):
        has_blockage = np.random.choice([0, 1])
        location = np.random.uniform(0.2, 0.8) if has_blockage else 0
        
        # صنع الإشارة بناءً على وجود انسداد من عدمه
        freq = 5
        clean = np.sin(2 * np.pi * freq * t) * np.exp(-20 * (t - 0.2)**2)
        if has_blockage:
            # إضافة صدى الانسداد
            echo = 0.4 * np.sin(2 * np.pi * freq * (t - location)) * np.exp(-40 * (t - location)**2)
            clean += echo
            
        dirty = clean + np.random.normal(0, 0.1, 500)
        X.append(dirty)
        Y.append([has_blockage, location])
        
    return np.array(X), np.array(Y)

# 2. بناء الموديل التشخيصي
X_diag, Y_diag = generate_diagnostic_data(3000)

radar_model = models.Sequential([
    layers.Input(shape=(500,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), # لمنع الحفظ الأعمى
    layers.Dense(64, activation='relu'),
    layers.Dense(2) # مخرجان: (نعم/لا) و (الموقع)
])

radar_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. التدريب على الكشف
print("Training the Radar to locate blockages...")
radar_model.fit(X_diag, Y_diag, epochs=30, verbose=0)

# 4. الاختبار النهائي (The Ultimate Test)
test_sample, test_label = generate_diagnostic_data(1)
prediction = radar_model.predict(test_sample)

print(f"\n--- [MONSTER REPORT] ---")
print(f"Actual - Blockage: {test_label[0][0]}, Location: {test_label[0][1]:.2f}")
print(f"Monster Prediction - Prob: {prediction[0][0]:.2f}, Location: {prediction[0][1]:.2f}")

# --- المرحلة الثالثة: نظام التشخيص الذكي ---
print("\n[STEP 3] Installing the Diagnostic Radar...")

# 1. دالة توليد بيانات مع "تسميات" (Labels) للانسداد
def generate_labeled_data(num_samples=2000):
    t = np.linspace(0, 1, 500)
    X = [] # الإشارة القذرة
    Y = [] # [هل يوجد انسداد؟ ، مكانه]
    
    for _ in range(num_samples):
        # 50% احتمال وجود انسداد
        has_blockage = np.random.choice([0, 1])
        location = np.random.uniform(0.3, 0.7) if has_blockage else 0
        
        # الموجة الأساسية
        clean = np.sin(2 * np.pi * 5 * t) * np.exp(-20 * (t - 0.2)**2)
        if has_blockage:
            # إضافة صدى الانسداد (Echo)
            echo = 0.5 * np.sin(2 * np.pi * 5 * (t - location)) * np.exp(-40 * (t - location)**2)
            clean += echo
            
        dirty = clean + np.random.normal(0, 0.12, 500)
        X.append(dirty)
        Y.append([has_blockage, location])
        
    return np.array(X), np.array(Y)

# 2. بناء "رادار التشخيص"
X_diag, Y_diag = generate_labeled_data(4000)

radar_model = models.Sequential([
    layers.Input(shape=(500,)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2), # لمنع "الحفظ" وضمان الفهم
    layers.Dense(128, activation='relu'),
    layers.Dense(2) # المخرجات: [احتمال الانسداد، الموقع]
])

radar_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. تدريب الرادار
print("Training the Radar to locate blockages through noise...")
radar_model.fit(X_diag, Y_diag, epochs=40, batch_size=32, verbose=0)

# 4. الاختبار النهائي (تقرير الوحش)
test_x, test_y = generate_labeled_data(1)
prediction = radar_model.predict(test_x)

print("\n" + "="*30)
print("      [MONSTER REPORT]")
print("="*30)
print(f"Target Reality -> Blockage: {test_y[0][0]}, Location: {test_y[0][1]:.2f}")
print(f"Monster Vision -> Detection: {prediction[0][0]:.2f}, Prediction: {prediction[0][1]:.2f}")
print("="*30)