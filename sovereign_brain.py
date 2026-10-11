import tensorflow as tf
from keras import layers, models

def build_sovereign_model(input_shape_acoustic=(500, 1), input_shape_env=(3,)):
    """
    بناء الهيكل العصبي البايزي للوحش:
    - مدخل 1: إشارات صوتية (Acoustic Signals)
    - مدخل 2: بيانات بيئية (حرارة، ضغط، لزوجة)
    """
    
    # [1] فرع معالجة الإشارة الصوتية (CNN Layer)
    acoustic_input = layers.Input(shape=input_shape_acoustic, name="Acoustic_Input")
    x = layers.Conv1D(32, kernel_size=3, activation='relu')(acoustic_input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    
    # [2] فرع البيانات البيئية (Dense Layer)
    env_input = layers.Input(shape=input_shape_env, name="Environmental_Input")
    y = layers.Dense(16, activation='relu')(env_input)
    
    # [3] عملية الدمج (Fusion Layer)
    combined = layers.concatenate([x, y])
    
    # [4] الطبقات البايزية لاتخاذ القرار تحت "عدم اليقين"
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(0.2)(z) # لضمان عدم الحفظ الصم (Generalization)
    
    # المخرجات: [احتمالية الانسداد، الموقع، معامل الثقة]
    output = layers.Dense(3, activation='sigmoid', name="Monster_Verdict")(z)
    
    model = models.Model(inputs=[acoustic_input, env_input], outputs=output)
    
    # تجميع الموديل بتركيز على تقليل خسائر الطاقة
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

if __name__ == "__main__":
    # اختبار سريع للهيكل
    m = build_sovereign_model()
    m.summary()
    print("✅ [BRAIN READY] هيكل الدماغ السيادي مكتمل.")