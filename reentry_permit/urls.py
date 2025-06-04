from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("generate_doc/", views.generate_doc, name="generate_doc"),
]