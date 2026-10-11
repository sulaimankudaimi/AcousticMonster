# industrial_plc_sim.py - UNIVERSAL VERSION
from pymodbus.server import StartTcpServer
import logging

# محاولة الاستيراد بأكثر من طريقة لكسر الدائرة المغلقة
try:
    from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
except ImportError:
    # في بعض إصدارات 3.x تم نقل السياق إلى مجلد فرعي
    from pymodbus.datastore import ModbusSequentialDataBlock
    from pymodbus.datastore.context import ModbusSlaveContext, ModbusServerContext

# إعداد البيانات الأساسية للوحش
initial_values = [45, 1, 1500] + [0]*97
data_block = ModbusSequentialDataBlock(0, initial_values)
context = ModbusServerContext(slaves=ModbusSlaveContext(hr=data_block, zero_mode=True), single=True)

print("--- ACOUSTIC MONSTER: PLC SERVER ---")
print("🚀 Status: RUNNING on localhost:5020")

if __name__ == "__main__":
    try:
        StartTcpServer(context=context, address=("127.0.0.1", 5020))
    except Exception as e:
        print(f"❌ Error: {e}")