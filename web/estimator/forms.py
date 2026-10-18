from django import forms

from .models import COST_DRIVER_FIELDS, COST_DRIVER_LABELS, Estimation, LocalProject


class NewEstimationForm(forms.Form):
    project_name = forms.CharField(label="Nombre del proyecto", max_length=200)
    kloc = forms.FloatField(label="Tamaño estimado (KLOC)", min_value=0.1)
    mode = forms.ChoiceField(label="Modo de desarrollo", choices=Estimation.mode.field.choices)
    rate_per_pm = forms.FloatField(label="Tarifa promedio (Bs/persona-mes)", min_value=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rating_choices = Estimation.rely.field.choices
        for field_name in COST_DRIVER_FIELDS:
            self.fields[field_name] = forms.ChoiceField(
                label=COST_DRIVER_LABELS[field_name],
                choices=rating_choices,
                initial="n",
            )

    def cost_driver_values(self):
        return {f: self.cleaned_data[f] for f in COST_DRIVER_FIELDS}


class LocalProjectForm(forms.ModelForm):
    class Meta:
        model = LocalProject
        fields = [
            "project_name", "institution", "kloc", "mode",
            *COST_DRIVER_FIELDS,
            "effort_real_pm", "cost_real_bs",
        ]
