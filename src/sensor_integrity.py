import numpy as np

class SensorShield:
    def __init__(self):
        self.sensor_names = ["Sensor_Alpha (Head)", "Sensor_Beta (Mid)", "Sensor_Gamma (Tail)"]
        self.health_status = {name: True for name in self.sensor_names}

    def perform_voting(self, predictions, uncertainties):
        """
        خوارزمية التصويت البايزي:
        تقوم بمقارنة التنبؤات واستبعاد الحساس الذي يبتعد عن المتوسط بمسافة كبيرة
        أو الذي يملك نسبة شك (Uncertainty) غير منطقية.
        """
        preds = np.array(predictions)
        uncerts = np.array(uncertainties)
        
        # 1. حساب المتوسط المرجح (الذي يثق أكثر بالحساسات ذات الشك المنخفض)
        weights = 1.0 / (uncerts + 1e-6)
        weighted_mean = np.average(preds, weights=weights)
        
        final_verdict = []
        for i, p in enumerate(preds):
            # إذا ابتعد الحساس عن المتوسط المرجح بأكثر من 25%، نعتبره "كاذباً" أو "معطلاً"
            deviation = abs(p - weighted_mean)
            if deviation > 0.25 or uncerts[i] > 0.5:
                self.health_status[self.sensor_names[i]] = False
            else:
                self.health_status[self.sensor_names[i]] = True
                final_verdict.append(p)
        
        # النتيجة النهائية هي متوسط الحساسات "الصادقة" فقط
        if not final_verdict:
            return weighted_mean, self.health_status # حالة طوارئ: الكل مشكوك فيه
        
        return np.mean(final_verdict), self.health_status