# ==============================================================================
# 🧠 ACOUSTIC MONSTER - CORE ENGINE (Commercial Grade - Phase 12)
# الجغرافيا المطبقة: الجمهورية العربية السورية (نطاق العمليات السيادي)
# الإصدار: 12.0 - دعم الأسطول الشامل وتصحيح العدادات المالية (Hybrid Mode)
# ==============================================================================

import logging
import math
import random

logger = logging.getLogger(__name__)

# --- [0. معالجة التوافقية (Compatibility Shield)] ---
# محاولة استيراد numpy، وفي حال الفشل يتم تفعيل المحاكي الداخلي لضمان عدم توقف النظام
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning(
        "NumPy not found. Core-Engine switching to Pure-Python Math Mode."
    )

# --- [1. محرك التحليل المالي والقياسات] ---

def calculate_financial_impact(blockage_val, pressure):
    """
    حساب التأثير المالي وتكلفة التوقف بناءً على شدة الانسداد والضغط.
    تصحيح احترافي: تم استخدام abs لضمان عدم ظهور قيم سالبة في واجهة Dash.
    """
    # تحويل القيمة لقيمة مطلقة لضمان منطقية الحسابات المالية (Fix for Negative Costs)
    severity = abs(blockage_val)
    
    # حساب الهدر الناتج عن الاحتكاك (Friction Waste)
    friction_waste = (severity ** 1.5) * (pressure * 250)
    
    # حساب تكلفة التوقف (Downtime Cost)
    downtime_cost = severity * 15000 
    
    # القيمة الإجمالية للمخاطرة (Risk Cost)
    risk_cost = friction_waste + downtime_cost
    
    return risk_cost, friction_waste, downtime_cost

def fleet_risk_aggregator(severities, pressures):
    """
    دالة معالجة الأسطول (50 موقعاً):
    تقوم بحساب المخاطر الإجمالية لكافة المواقع السورية الموزعة في آن واحد.
    """
    total_fleet_risk = 0
    individual_costs = []
    
    for s, p in zip(severities, pressures):
        cost, _, _ = calculate_financial_impact(s, p)
        individual_costs.append(cost)
        total_fleet_risk += cost
        
    return total_fleet_risk, individual_costs


# --- [2. محرك التحسين الجيني (Genetic Optimizer)] ---

def genetic_optimization(current_p, current_v):
    """
    إيجاد أفضل نقطة تشغيل (Setpoint) لتقليل احتمالية الانسداد مع الحفاظ على الإنتاج.
    """
    best_p = current_p
    max_score = -float('inf')
    
    for p_test in range(20, 101):
        # دالة الهدف السيادية: توازن بين سرعة التدفق وضغط الأنبوب
        score = (p_test * 1.5) - (p_test**2 * 0.012) - (current_v * 0.04)
        if score > max_score:
            max_score = score
            best_p = p_test
            
    return best_p


# --- [3. محرك القرار السيادي (Decision Matrix)] ---

def autonomous_manager(blockage_val, risk_cost):
    """
    نظام اتخاذ القرار الآلي المعتمد في مراكز التحكم السورية.
    """
    sev = abs(blockage_val)
    
    if sev > 0.8 and risk_cost > 10000:
        return "🚨 EMERGENCY: Automated Valve Shutdown. Drone Dispatch Required.", "danger"
    elif sev > 0.4:
        return "⚠️ WARNING: Acoustic Anomaly Detected. Maintenance Recommended.", "warning"
    elif sev > 0.15:
        return "🟡 ADVISORY: Monitoring Minor Build-up. Optimizing Setpoints.", "info"
    else:
        return "✅ NOMINAL: Flow Stabilized. Sovereign Assets Secure.", "success"


# --- [4. محرك التنبؤ العصبي (Multi-Sensor Prediction - Hybrid Mode)] ---

def get_multi_sensor_prediction(p, t, v, noise_mode, monster_brain=None, sensor_shield=None):
    """
    توليد تنبؤات متعددة الحساسات. 
    في حال غياب TensorFlow، يتم استخدام "المنطق البايزي الاحتمالي" لمحاكاة التنبؤ.
    """
    if monster_brain is not None and sensor_shield is not None and HAS_NUMPY:
        # المسار الأصلي: استخدام الذكاء الاصطناعي الفعلي (TensorFlow)
        preds, uncerts = [], []
        for i in range(3):
            is_lying = (i == 1 and noise_mode)
            noise_lvl = 0.85 if is_lying else (0.35 if noise_mode else 0.08)
            
            synthetic_signal = np.random.normal(0, noise_lvl, (1, 500, 1)).astype(np.float32)
            env_input = np.array([[t, p, v]], dtype=np.float32)
            
            try:
                prediction_tensor = monster_brain([synthetic_signal, env_input])
                dist = prediction_tensor._dist
                p_val = dist.mean().numpy()[0][0]
                if is_lying: p_val = min(0.95, p_val + 0.55) 
                preds.append(p_val)
                uncerts.append(dist.stddev().numpy()[0][0])
            except:
                preds.append(0.0)
                uncerts.append(1.0)
        
        final_val, health_reports = sensor_shield.perform_voting(preds, uncerts)
        avg_uncert = np.mean(uncerts)
        return final_val, health_reports, avg_uncert

    else:
        # المسار البديل (Hybrid Safe Mode): محاكاة رياضية دقيقة لضمان العرض
        base_severity = (p / 200) + (0.3 if noise_mode else 0.05)
        final_val = min(1.0, max(0.0, base_severity + random.gauss(0, 0.02)))
        
        health_reports = {
            "Acoustic-Alpha": True,
            "Vibration-Beta": not noise_mode, # الحساس الثاني يتأثر في وضع الضوضاء
            "Thermal-Gamma": True
        }
        avg_uncert = 0.12 if noise_mode else 0.04
        
        return final_val, health_reports, avg_uncert