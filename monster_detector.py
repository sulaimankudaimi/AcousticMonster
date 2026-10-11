import numpy as np
import matplotlib.pyplot as plt
import os

# إعداد نافذة العرض لتكون تفاعلية
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.canvas.manager.set_window_title('Sovereign P.12 - Acoustic Monster Vision')

def monster_decision_unit(signal, env_data, model, original_packet):
    """
    تحليل الإشارة الحية، إصدار الحكم، ورسم النبضات الفيزيائية للأنبوب.
    """
    try:
        # 1. التنبؤ عبر الدماغ السيادي (Inference)
        # الإدخال: [Signal (1,500,1), Env (1,3)]
        prediction = model.predict([signal, env_data], verbose=0)
        
        # 2. تفكيك مخرجات الذكاء الاصطناعي
        # الموديل يخرج: [احتمالية الانسداد، الموقع، معامل اليقين]
        prob_blocked = float(prediction[0][0])
        detected_loc = float(prediction[0][1]) * 500  # تحويل الموقع النسبي لطول الأنبوب
        confidence = float(prediction[0][2])
        
        # 3. استخراج البيانات الوصفية والمالية
        header = original_packet.get("header", {})
        physics = original_packet.get("physics_vector", {})
        financial = original_packet.get("financial_engine", {"current_risk_exposure": 0})
        
        # 4. إصدار الحكم النهائي (Verdict)
        is_blocked = prob_blocked > 0.7  # عتبة الحساسية السيادية
        
        # 5. التحديث المرئي (Visual Update)
        update_visuals(signal.flatten(), prob_blocked, detected_loc, physics)

        # 6. التقرير النصي في الـ Terminal
        render_terminal_report(header, is_blocked, detected_loc, confidence, financial)

    except Exception as e:
        print(f"❌ [DETECTOR CRASH] Error: {str(e)}")

def update_visuals(signal_data, prob, loc, physics):
    """تحديث الرسوم البيانية الحية لنبضات الأنبوب"""
    ax1.clear()
    ax2.clear()

    # الرسم الأول: النبضة الصوتية الخام
    ax1.plot(signal_data, color='#00ff41', linewidth=1) # اللون الأخضر الرقمي
    ax1.set_title(f"Acoustic Pulse Integrity | Pressure: {physics.get('pressure', 0)} Pa", color='white')
    ax1.set_facecolor('black')
    ax1.grid(True, alpha=0.2)

    # الرسم الثاني: خريطة احتمالية الانسداد
    x_axis = np.linspace(0, 500, 500)
    # رسم منطقة خطر إذا تجاوزت الاحتمالية 70%
    if prob > 0.7:
        ax2.axvline(x=loc, color='red', linestyle='--', label='Potential Blockage')
        ax2.fill_between(x_axis, 0, prob, color='red', alpha=0.3)
    else:
        ax2.fill_between(x_axis, 0, prob, color='cyan', alpha=0.3)

    ax2.set_title(f"Blockage Probability: {prob*100:.1f}%", color='white')
    ax2.set_ylim(0, 1)
    ax2.set_facecolor('black')
    ax2.legend()

    plt.tight_layout()
    fig.patch.set_facecolor('#121212')
    plt.pause(0.01) # تحديث سريع جداً لضمان الانسيابية

def render_terminal_report(header, is_blocked, loc, conf, financial):
    """طباعة التقرير الاحترافي في نافذة الأوامر"""
    color_code = "\033[91m" if is_blocked else "\033[92m" # أحمر للخطر، أخضر للسلامة
    reset = "\033[0m"
    
    print("\n" + "═"*65)
    print(f"🕵️ [MONSTER ANALYSIS] Time: {header.get('timestamp', 'N/A')}")
    
    if is_blocked:
        print(f"{color_code}🚨 STATUS: [BLOCKAGE DETECTED] !!!{reset}")
        print(f"📍 LOCATION: {loc:.2f} Meters along the P.12 pipeline")
        print(f"⚖️ CONFIDENCE: {conf*100:.2f}%")
        print(f"💰 FINANCIAL RISK: ${financial['current_risk_exposure']:,.2f} USD/hr")
    else:
        print(f"{color_code}✅ STATUS: [PIPELINE CLEAR]{reset}")
        print(f"🛡️ INTEGRITY: Flow is Laminar & Stable")
        print(f"📊 SYSTEM SYNC: Confidence {conf*100:.2f}%")
        
    print("═"*65 + "\n")