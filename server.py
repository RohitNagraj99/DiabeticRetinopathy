import pickle
from flask import Flask, request, jsonify
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


def prepareImage(img):
    img = np.array(img, dtype=np.float32)
    img = img/255.0
    rgb_weights = [0.2989, 0.5870, 0.1140]
    img = np.dot(img, rgb_weights)
    img = img.reshape(1, -1)
    return img


@app.route('/', methods=['POST'])
def predict():
    img = plt.imread(request.files['file'])
    img = prepareImage(img)
    labels = {0: 'No Diabetic Retinopathy', 1: 'Mild Diabetic Retinopathy',
              2: 'Moderate Diabetic Retinopathy', 3: 'Severe Diabetic Retinopathy',
              4: 'Proliferate Diabetic Retinopathy'}
    pred = model.predict(img)
    return labels[int(pred)]


if __name__ == '__main__':
    app.run(debug=True)
