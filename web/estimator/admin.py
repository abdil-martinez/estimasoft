from django.contrib import admin

from .models import Estimation, LocalProject


@admin.register(Estimation)
class EstimationAdmin(admin.ModelAdmin):
    list_display = ["project_name", "kloc", "mode", "effort_mlp_pm", "effort_cocomo_pm", "cost_bs", "created_at"]
    list_filter = ["mode"]
    search_fields = ["project_name"]


@admin.register(LocalProject)
class LocalProjectAdmin(admin.ModelAdmin):
    list_display = ["project_name", "institution", "kloc", "effort_real_pm", "effort_mlp_pm", "created_at"]
    list_filter = ["mode"]
    search_fields = ["project_name", "institution"]
