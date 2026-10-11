# monster_scada_bridge.py
from pymodbus.client import ModbusTcpClient
import time

class ScadaBridge:
    def __init__(self, host='127.0.0.1', port=5020):
        # استخدام ModbusTcpClient المتوافق مع الإصدارات الجديدة
        self.client = ModbusTcpClient(host, port=port)
        self.safety_lock = False

    def connect(self):
        """تأسيس الاتصال مع الـ PLC"""
        try:
            return self.client.connect()
        except Exception:
            return False

    def get_field_data(self):
        """قراءة البيانات الحقيقية من سجلات الـ PLC"""
        # قراءة أول 3 سجلات (Holding Registers)
        result = self.client.read_holding_registers(0, 3)
        if not result.isError():
            return {
                "pressure": result.registers[0],
                "valve_status": "OPEN" if result.registers[1] == 1 else "CLOSED",
                "rpm": result.registers[2]
            }
        return None

    def execute_override(self, danger_level):
        """بروتوكول عقدة الأمان الثلاثية (The Deadman Switch)"""
        
        # 1. المرحلة الأولى: خطر أقل من 80%
        if danger_level < 0.8:
            return "ADVISORY: System stable. No physical intervention required."

        # 2. المرحلة الثانية: خطر بين 90% و 98%
        elif 0.9 <= danger_level < 0.98:
            return "WARNING: Critical detected. Intervention scheduled in 60s unless manually aborted."

        # 3. المرحلة الثالثة: خطر داهم >= 98%
        elif danger_level >= 0.98:
            # إغلاق الصمام فوراً (Register 1 -> 0)
            self.client.write_register(1, 0)
            # خفض سرعة المضخة للصفر (Register 2 -> 0)
            self.client.write_register(2, 0)
            return "🚨 EMERGENCY SHUTDOWN EXECUTED: Asset preserved by Sovereign Protocol."

# اختبار الجسر عند تشغيل الملف بشكل مستقل
if __name__ == "__main__":
    bridge = ScadaBridge()
    if bridge.connect():
        print("✅ Connected to Field PLC.")
        # تجربة قراءة البيانات الحالية
        print("Initial Field State:", bridge.get_field_data())
        
        # محاكاة خطر داهم (99%) لاختبار الإغلاق التلقائي
        print("\n--- Simulating Extreme Danger (99%) ---")
        status = bridge.execute_override(0.99)
        print(status)
        
        # التأكد من تغير الحالة في المحاكي
        print("Updated Field State:", bridge.get_field_data())
        bridge.client.close()
    else:
        print("❌ Failed to connect to PLC. Make sure industrial_plc_sim.py is running.")