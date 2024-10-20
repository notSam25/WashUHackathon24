from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
from model.src.predict import predict_from_dataframe


class ESIPredictionView(APIView):
    def options(self, request, *args, **kwargs):
        response = Response(status=status.HTTP_200_OK)
        return response

    def post(self, request) -> Response:
        print(len(request.data))
        try:
            input_data = pd.DataFrame([request.data])
            print(input_data)
            esi_scores, probabilities = predict_from_dataframe(input_data)
            print(esi_scores)
            response = {
                "esi_scores": esi_scores.tolist(),
                "probabilities": [prob.tolist() for prob in probabilities],
            }
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)