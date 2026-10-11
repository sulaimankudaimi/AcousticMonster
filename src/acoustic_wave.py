import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # أضفنا هذا السطر لاستخدامه في الشروط

# 1. تعريف النطاق المكاني (طول الأنبوب) والزماني
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 2. تعريف معادلة الموجة الصوتية مع سرعة متغيرة (تمثيل الانسداد)
def wave_equation(x, y):
    space_x = x[:, 0:1]
    c_nominal = 1.0
    
    # التعديل هنا: نستخدم tf.where بدلاً من dde.backend.where
    c = tf.where(space_x > 0.4, 0.5, c_nominal)
    
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    
    return dy_tt - c**2 * dy_xx

# 3. الشروط الحدودية
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

# 4. الشرط الابتدائي
def initial_condition(x):
    return np.exp(-100 * (x[:, 0:1])**2)

ic = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)

# 5. بناء الموديل
data = dde.data.TimePDE(geomtime, wave_equation, [bc, ic], num_domain=2000, num_boundary=100, num_initial=100)
net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# 6. التدريب
model.compile("adam", lr=0.001)
print("Starting Acoustic Monster Training (Obstacle Mode)...")
losshistory, train_state = model.train(iterations=5000)

# 7. التنبؤ ورسم النتائج
x_points = np.linspace(-1, 1, 200)
t_points = np.linspace(0, 1, 200)
X, T = np.meshgrid(x_points, t_points)
test_points = np.vstack([X.ravel(), T.ravel()]).T

u_pred = model.predict(test_points)
U = u_pred.reshape(200, 200)

# الرسم البياني
plt.figure(figsize=(10, 6))
plt.pcolormesh(T, X, U, shading='auto', cmap='jet')
plt.colorbar(label='Acoustic Pressure (p)')
plt.axhline(y=0.4, color='white', linestyle='--', label='Obstacle Start')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('Acoustic Monster: Impact of Obstacle on Wave Propagation')
plt.legend()

print("\n[SUCCESS] Training finished. Displaying the plot...")
plt.show(block=True)