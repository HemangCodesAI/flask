from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

def predict(image):
    resized_image = image.resize((32, 32))
    img_array = np.array(resized_image.convert('RGB'))
    normalized_image = img_array / 255.0
    model = load_model("achawala.h5")
    ans = model.predict(normalized_image.reshape(1, 32, 32, 3))
    if ans[0][0] < ans[0][1]:
        return "Non-Anemic"
    else:
        return "Anemic"

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        file = request.files['file']
        image = Image.open(file)
        prediction_result = predict(image)
        return print({"prediction_result": prediction_result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
