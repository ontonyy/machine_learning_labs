import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

dir = 'random_images'
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
model = load_model("image_cnn_model.h5")

image_path = input("Input image for prediction(with format .jpg/.png/.jpeg): ")

def get_top_predictions(prediction):
    top_probs_indices = np.argsort(prediction)[0][::-1][:10]
    top_probs = prediction[0][top_probs_indices]
    top_class_names = [class_names[i] for i in top_probs_indices]
    return top_class_names, top_probs


plot_img = cv.imread(f"{image_path}")
img = plot_img.astype(float)

img = cv.resize(img, (32, 32))
img = np.expand_dims(img, axis=0) / 255.0

prediction = model.predict(img)
index = np.argmax(prediction)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

ax1.set_title(f"Truly is {image_path.split('.')[0]}\nPrediction is {class_names[index]}")
ax1.imshow(cv.cvtColor(plot_img, cv.COLOR_BGR2RGB))
ax1.axis("off")

top_names, top_predictions = get_top_predictions(prediction)
ax2.barh(top_names, top_predictions)
ax2.set_xlabel("Probability")
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.show()
