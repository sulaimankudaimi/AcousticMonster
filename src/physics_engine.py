import deepxde as dde
import numpy as np
import tensorflow as tf

# --- [1. محرك فيزياء الوسط الحقيقي] ---
def get_fluid_properties(T, P):
    """
    حساب سرعة الصوت بناءً على الحرارة والضغط (نموذج النفط الخام)
    T: Celsius, P: Bar
    """
    # سرعة الصوت تتأثر بالحرارة (تنقص) وبالضغط (تزيد)
    c = 1500 - 3.5 * T + 0.05 * P 
    # معامل اللزوجة الحركية (يزداد مع البرودة ويؤثر على التخميد)
    viscosity = 0.1 * np.exp(500 / (T + 273)) 
    return c, viscosity

# --- [2. تعريف النطاق المكاني والزماني] ---
# أنبوب بطول 10 متر، محاكاة لمدة 0.1 ثانية
geom = dde.geometry.Interval(0, 10)
timedomain = dde.geometry.TimeDomain(0, 0.1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# --- [3. معادلة الموجة الفيزيائية السيادية (PINN)] ---
def pde(x, y):
    """
    تعريف معادلة الموجة مع اعتبار اللزوجة والتخميد
    y: هو ضغط الموجة p(x, t)
    """
    # مدخلات افتراضية للبيئة أثناء حل الـ PDE
    T_ref, P_ref = 45.0, 50.0  # بيئة صحراوية حارة وضغط عالٍ
    c_val, eta = get_fluid_properties(T_ref, P_ref)
    
    # المشتقات
    p_tt = dde.grad.hessian(y, x, i=1, j=1) # المشتقة الثانية للزمن
    p_xx = dde.grad.hessian(y, x, i=0, j=0) # المشتقة الثانية للمكان
    p_t = dde.grad.jacobian(y, x, i=0, j=1) # المشتقة الأولى للزمن (للتخميد)

    # معادلة الموجة مع التخميد اللزوجي: p_xx - (1/c^2)p_tt - eta*p_t = 0
    return p_xx - (1 / c_val**2) * p_tt - (eta * 0.01) * p_t

# --- [4. الشروط الحدودية (النبضة المرسلة)] ---
def input_pulse(x):
    # نبضة جيبية عند بداية الأنبوب (x=0) عند الزمن t=0
    return np.sin(2 * np.pi * 100 * x[:, 1:2]) * np.exp(-100 * (x[:, 1:2] - 0.01)**2)

ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)
bc_source = dde.icbc.DirichletBC(geomtime, input_pulse, lambda x, on_boundary: on_boundary and np.isclose(x[0], 0))
bc_end = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda x, on_boundary: on_boundary and np.isclose(x[0], 10))

# --- [5. بناء الموديل وتجميعه] ---
data = dde.data.TimePDE(
    geomtime, pde, [bc_source, bc_end, ic], 
    num_domain=2000, num_boundary=100, num_initial=100
)

# شبكة عصبية أعمق لمعالجة تعقيد الفيزياء (5 طبقات، 64 عصبون)
net = dde.nn.FNN([2] + [64] * 5 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)

# --- [6. التدريب السيادي] ---
print("🚀 Training the Physics Engine in Harsh Environments...")
model.compile("adam", lr=1e-3)
model.train(iterations=5000)

# --- [7. دالة توليد البيانات للتدريب اللاحق] ---
def generate_synthetic_harsh_data(num_samples=1000):
    """توليد مصفوفة بيانات تحتوي على [إشارة، حرارة، ضغط، لزوجة]"""
    X_test = geomtime.random_points(num_samples)
    y_pred = model.predict(X_test)
    
    # إضافة ضجيج عشوائي يحاكي الواقع
    noise = np.random.normal(0, 0.02, y_pred.shape)
    return X_test, y_pred + noise

if __name__ == "__main__":
    X, Y = generate_synthetic_harsh_data()
    print(f"[SUCCESS] Physics Engine Ready. Generated {len(X)} points of data.")