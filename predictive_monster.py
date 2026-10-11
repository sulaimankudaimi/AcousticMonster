import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. توليد بيانات تاريخية (تراكم الترسبات عبر 30 يوم)
def generate_growth_history(days=60):
    t = np.linspace(0, days, days)
    # نمو تراكمي (Linear + Exponential) مع ضجيج
    growth_rate = 0.015 
    history = growth_rate * t + 0.002 * t**1.5 + np.random.normal(0, 0.05, days)
    history = np.clip(history, 0, 1) # النسبة بين 0 و 1 (100% انسداد)
    return t, history

# 2. تحضير البيانات لـ LSTM (نستخدم نافذة من 7 أيام لتوقع اليوم التالي)
def create_sequences(data, window=7):
    x, y = [], []
    for i in range(len(data) - window):
        x.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(x).reshape(-1, window, 1), np.array(y)

days_total = 100
t, full_history = generate_growth_history(days_total)
train_data = full_history[:60] # تدريب على أول 60 يوم

window_size = 7
X, Y = create_sequences(train_data, window_size)

# 3. بناء نموذج LSTM التنبؤي
model = models.Sequential([
    layers.Input(shape=(window_size, 1)),
    layers.LSTM(50, activation='relu', return_sequences=False),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

print("Training the Predictive Brain (LSTM)...")
model.fit(X, Y, epochs=100, verbose=0)

# 4. التنبؤ بالمستقبل (الأسبوعين القادمين)
future_steps = 14
current_window = train_data[-window_size:].tolist()
predictions = []

for _ in range(future_steps):
    input_window = np.array(current_window[-window_size:]).reshape(1, window_size, 1)
    next_val = model.predict(input_window, verbose=0)[0][0]
    predictions.append(next_val)
    current_window.append(next_val)

# 5. حساب الـ ROI اللحظي والخسائر
current_clog = train_data[-1]
predicted_clog_14d = predictions[-1]
daily_loss_constant = 5000 # خسارة 5000 دولار يومياً عند الانسداد الكامل
estimated_loss = (predicted_clog_14d - current_clog) * daily_loss_constant * 14

# --- الرسم البياني التنبؤي ---
plt.figure(figsize=(12, 6))
plt.plot(t[:60], train_data, label="Historical Data (Last 60 Days)", color='blue')
plt.plot(t[60:60+future_steps], predictions, '--', label="Monster Forecast (Next 14 Days)", color='red')
plt.axhline(y=0.8, color='orange', linestyle=':', label="Critical Threshold (Action Required)")
plt.title("Acoustic Monster: Predictive Maintenance & ROI Analysis")
plt.xlabel("Days")
plt.ylabel("Blockage Severity (0.0 to 1.0)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n--- [ECONOMICS REPORT] ---")
print(f"Current Blockage: {current_clog*100:.1f}%")
print(f"Predicted Blockage in 14 days: {predicted_clog_14d*100:.1f}%")
print(f"⚠️ POTENTIAL REVENUE LOSS: ${estimated_loss:,.2f} if no maintenance is performed.")
print(f"DECISION: {'URGENT CLEANING REQUIRED' if predicted_clog_14d > 0.7 else 'Monitor Closely'}")