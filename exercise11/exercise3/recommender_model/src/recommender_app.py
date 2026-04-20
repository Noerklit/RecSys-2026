from flask import Flask, request, jsonify
from surprise import dump

app = Flask(__name__)

# IMPORTANT: Use the absolute path as discussed for Docker
model_path = '/models/model_SVD.pkl'
_, algo = dump.load(model_path)

@app.route("/", methods=["GET"])
def predict():
    # The simulator sends data in the request body
    data = request.form if request.form else request.json
    
    # Use .get() to avoid KeyErrors if the simulator hasn't started yet
    u_id = data.get('userid')
    i_id = data.get('itemid')
    
    if u_id is None or i_id is None:
        return jsonify({"error": "Missing data"}), 400

    prediction = algo.predict(str(u_id), str(i_id))
    
    # The simulator expects 'estimated_rating' as the key
    return jsonify({
        "estimated_rating": prediction.est
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)