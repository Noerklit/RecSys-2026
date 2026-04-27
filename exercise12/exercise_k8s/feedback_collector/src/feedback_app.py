from flask import Flask, request, jsonify
import numpy as np
import os
import pandas as pd
import logging

app = Flask(__name__)
errors = []

# So our logs are not flooded with all the GET requests
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route("/", methods=["GET"])
def collect():
    # Extracting from request.form because 'requests.get(url, data=...)' 
    # places data in the body, not args.
    data = request.form
    
    actual = float(data.get('rating'))
    est = float(data.get('estimated_rating'))
    
    # Keep track in memory
    error_sq = (actual - est) ** 2
    errors.append(error_sq)
    rmse = np.sqrt(np.mean(errors))
    
    print(f"Received tuple containing actual: {actual} and estimated: {est}")
    
    print(f"Current RMSE is: {rmse}")
    return jsonify({"status": "ok", "rmse": rmse})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)