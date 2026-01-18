import tensorflow as tf
from tensorflow import keras

# 1) حمّل الموديل الأصلي (هنا TFSMLayer مسموح)
model_with_tfsm = keras.models.load_model(
    "model.h5",
    compile=False
)

# 2) خذ الـ input والـ output النهائي فقط
inputs = model_with_tfsm.input
outputs = model_with_tfsm.output

# 3) أنشئ موديل Keras نظيف
clean_model = keras.Model(inputs=inputs, outputs=outputs)

# 4) احفظه بدون أي TFSMLayer
clean_model.save("clean_model.h5")

print("✅ clean_model.h5 created (NO TFSMLayer)")
