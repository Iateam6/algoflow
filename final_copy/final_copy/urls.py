from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("generate-final-copy/", views.final_copy, name="final_copy"),
]