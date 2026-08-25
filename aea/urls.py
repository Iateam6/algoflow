from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("generations/", views.generate_doc, name="generate_doc"),
    path("generate_doc/", views.create_generation, name="create_generation"),
    path("generate_doc/<uuid:job_id>/download/", views.download_generation, name="download_generation"),
]
