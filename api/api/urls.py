from django.urls import path
from .views import ESIPredictionView

urlpatterns = [
    path("predict/", ESIPredictionView.as_view(), name="esi-prediction"),
]
