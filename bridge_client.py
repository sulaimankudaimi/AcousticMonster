import asyncio
import websockets
import json
import os
import hmac
import hashlib
import traceback
from cryptography.fernet import Fernet

class SovereignBridgeClient:
    """
    الجسر السيادي: المسؤول عن استقبال، فك تشفير، والتحقق من سلامة 
    بيانات التوأم الرقمي P.12 ونقلها إلى محرك التحليل.
    """
    def __init__(self, key_path=r"F:\AcousticMonster\models\sovereign_vault.key"):
        self.uri = "ws://127.0.0.1:8000/ws/stream"
        self.ip_secret = "SOVEREIGN_SECRET_2026"  # كود الحماية المشترك
        
        # [1] تأمين مفتاح التشفير
        try:
            if not os.path.exists(key_path):
                raise FileNotFoundError(f"المفتاح غير موجود في: {key_path}")
                
            with open(key_path, "rb") as f:
                self.key = f.read().strip()
            
            self.cipher_suite = Fernet(self.key)
            print(f"✅ [SECURITY] Sovereign Key Loaded. Bridge Security: AES-256")
        except Exception as e:
            print(f"❌ [CRITICAL] Key Initialization Failed: {e}")
            raise

    def verify_ip_signature(self, packet):
        """التحقق من التوقيع الرقمي لمنع التلاعب بالبيانات (IP Integrity)"""
        try:
            header = packet.get("header", {})
            signature_received = header.get("ip_signature", "")
            
            # إعادة بناء مصفوفة التحقق (يجب أن تطابق تماماً ما يتم في الملف الأول)
            timestamp = str(header.get("timestamp"))
            location = str(header.get("location_id"))
            
            data_to_verify = f"{timestamp}-{location}".encode()
            
            expected_sig_hash = hmac.new(
                self.ip_secret.encode(),
                data_to_verify,
                hashlib.sha256
            ).hexdigest()
            
            expected_signature = f"SIG-P12-{expected_sig_hash[:16].upper()}"
            
            return hmac.compare_digest(expected_signature, signature_received)
        except Exception:
            return False

    async def start_bridge(self, brain_callback):
        """فتح قناة الاتصال المستمرة مع التوأم الرقمي"""
        print(f"🛰️ [BRIDGE] Connecting to Sovereign Stream at {self.uri}...")
        
        retry_delay = 5
        while True:
            try:
                async with websockets.connect(self.uri, ping_interval=10, ping_timeout=20) as websocket:
                    print("✅ [BRIDGE ACTIVE] الجسر متصل الآن بالواقع الفيزيائي.")
                    
                    while True:
                        # 1. استقبال الحزمة الخام
                        raw_payload = await websocket.recv()
                        
                        try:
                            # 2. فك التشفير السيادي
                            # تحويل النص المستلم إلى Bytes إذا لزم الأمر
                            if isinstance(raw_payload, str):
                                raw_payload = raw_payload.encode()
                                
                            decrypted_data = self.cipher_suite.decrypt(raw_payload).decode()
                            packet = json.loads(decrypted_data)
                            
                            # 3. التدقيق الأمني (IP Signature Check)
                            if self.verify_ip_signature(packet):
                                # 4. النجاح: تمرير البيانات الموثوقة لـ main_fusion.py
                                await brain_callback(packet)
                            else:
                                print("⚠️ [SECURITY ALERT] حزمة بيانات مشبوهة! التوقيع غير متطابق.")
                                
                        except Exception as e:
                            print(f"⚠️ [PROCESSING ERROR] فشل فك التشفير أو التحليل: {e}")
                            # طباعة تتبع الخطأ في حالة الحاجة للتدقيق العميق
                            # traceback.print_exc() 

            except (websockets.ConnectionClosed, ConnectionRefusedError):
                print(f"🔌 [DISCONNECTED] الجسر مقطوع. إعادة المحاولة خلال {retry_delay} ثوانٍ...")
                await asyncio.sleep(retry_delay)
            except Exception as e:
                print(f"❌ [UNKNOWN ERROR] تعطل الجسر: {e}")
                await asyncio.sleep(retry_delay)

# --- محاكاة التشغيل المستقل (للإختبار فقط) ---
async def debug_callback(packet):
    p = packet["physics_vector"]
    print(f"📊 [LIVE DATA] P:{p['pressure']} | V:{p['velocity']} | L:{p['physics_loss']}")

if __name__ == "__main__":
    client = SovereignBridgeClient()
    try:
        asyncio.run(client.start_bridge(debug_callback))
    except KeyboardInterrupt:
        print("\n🛑 Bridge manually stopped.")