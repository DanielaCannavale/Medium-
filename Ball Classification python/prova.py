from PIL import Image
import numpy as np
import onnxruntime as ort
onnx_path = r"C:\Users\Daniela\PycharmProjects\Ball\Ball-Classification\model.onnx"
img = Image.open("test.jpg").convert('RGB').resize((224,224))
img=np.expand_dims(np.asarray(img, dtype="float32"),axis=0)
img=img/255

sess_ort = ort.InferenceSession(onnx_path,providers=ort.get_available_providers())

outputs = sess_ort.run(None, {sess_ort.get_inputs()[0].name: img})
print(outputs)
