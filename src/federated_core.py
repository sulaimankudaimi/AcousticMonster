import numpy as np
import tensorflow as tf
import os
import pickle
from cryptography.fernet import Fernet

class FederatedCoordinator:
    def __init__(self, model):
        self.model = model
        self.global_weights_path = "models/global_monster_weights.npy"
        self.secure_vault_path = "models/sovereign_vault.key"
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)

    def _load_or_generate_key(self):
        """إدارة مفاتيح التشفير AES-256 للمرحلة الثامنة"""
        if os.path.exists(self.secure_vault_path):
            with open(self.secure_vault_path, "rb") as key_file:
                return key_file.read()
        else:
            key = Fernet.generate_key()
            with open(self.secure_vault_path, "wb") as key_file:
                key_file.write(key)
            return key

    def get_local_weights(self):
        """استخراج الأوزان الحالية من النموذج المحلي"""
        return self.model.get_weights()

    def encrypt_and_save_weights(self, weights):
        """المرحلة 8: تشفير الأوزان قبل حفظها في الخزانة السيادية"""
        serialized_weights = pickle.dumps(weights)
        encrypted_data = self.cipher.encrypt(serialized_weights)
        
        vault_file = self.global_weights_path.replace(".npy", ".vault")
        with open(vault_file, "wb") as f:
            f.write(encrypted_data)
        return True

    def simulate_global_merging(self, local_weights):
        """
        المرحلة 9: محاكاة ذكاء الأساطيل (Fleet Intelligence)
        دمج الأوزان المحلية مع قراءات قادمة من مواقع (Basra, Kirkuk, Dahuk)
        """
        # توليد مصفوفات أوزان وهمية تمثل متوسط ذكاء السرب العالمي
        swarm_wisdom = []
        for w in local_weights:
            # إضافة تذبذب بسيط (Noise) لتمثيل اختلاف البيانات في المواقع الأخرى
            remote_node_1 = w + np.random.normal(0, 0.015, w.shape)
            remote_node_2 = w + np.random.normal(0, 0.025, w.shape)
            
            # خوارزمية FedAvg المرجحة: (أوزاني + أوزان العقدة 1 + أوزان العقدة 2) / 3
            averaged_w = (w + remote_node_1 + remote_node_2) / 3.0
            swarm_wisdom.append(averaged_w.astype(np.float32))
            
        return swarm_wisdom

    def apply_sync(self, new_weights):
        """تحديث دماغ الوحش بالأوزان العالمية وتأمينها"""
        try:
            # تحديث النموذج في الذاكرة الحية
            self.model.set_weights(new_weights)
            
            # المرحلة 8: التشفير الفيزيائي للأوزان الجديدة
            self.encrypt_and_save_weights(new_weights)
            
            # حفظ نسخة احتياطية (Unencrypted) للتوافق السريع
            np.save(self.global_weights_path, np.array(new_weights, dtype=object))
            return True
        except Exception as e:
            print(f"Sync Error: {e}")
            return False

    def get_fleet_status(self):
        """محاكاة بيانات الأسطول للمرحلة التاسعة"""
        sites = ['Basra_Gate_1', 'Kirkuk_Alpha', 'Dahuk_North', 'Baghdad_Hub']
        return [
            {'site': sites[0], 'blockage': 0.12, 'health': 100, 'status': 'Secure'},
            {'site': sites[1], 'blockage': 0.65, 'health': 85, 'status': 'Warning'},
            {'site': sites[2], 'blockage': 0.05, 'health': 98, 'status': 'Secure'},
            {'site': sites[3], 'blockage': 0.88, 'health': 40, 'status': 'Stressed'}
        ]