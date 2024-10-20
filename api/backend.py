from http.server import BaseHTTPRequestHandler
import pandas as pd
import json

from ..model.src.predict import predict_from_dataframe


class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        # Step 1: Get the length of the data
        content_length = int(self.headers["Content-Length"])

        # Step 2: Read and parse the incoming JSON data from the request
        post_data = self.rfile.read(content_length)
        input_json = json.loads(post_data)

        # Step 3: Convert JSON data to a Pandas DataFrame
        input_data = pd.DataFrame([input_json])

        # Step 4: Get ESI predictions and probabilities
        esi_scores, probabilities = predict_from_dataframe(input_data)

        # Step 5: Prepare the response
        response = {
            "esi_scores": esi_scores.tolist(),
            "probabilities": [prob.tolist() for prob in probabilities],
        }

        # Step 6: Send response
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))
        return
