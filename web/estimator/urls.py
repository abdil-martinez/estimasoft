from django.urls import path

from . import views

app_name = "estimator"

urlpatterns = [
    path("", views.panel, name="panel"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("nueva-estimacion/", views.new_estimation, name="new_estimation"),
    path("modelo-y-evaluacion/", views.model_eval, name="model_eval"),
    path("proyectos-locales/", views.local_projects, name="local_projects"),
]
