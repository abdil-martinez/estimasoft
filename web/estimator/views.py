from django.contrib import messages
from django.shortcuts import redirect, render

from . import ml
from .charts import build_loss_curve, build_scatter
from .dashboard_charts import build_effort_scatter, build_local_error_bars, build_mode_donut
from .forms import LocalProjectForm, NewEstimationForm
from .models import COST_DRIVER_FIELDS, MODE_CHOICES, Estimation, LocalProject


def _model_unavailable_message():
    return (
        "El modelo todavia no esta entrenado. Corre scripts/train_eval.py y "
        "scripts/train_final_model.py desde la raiz del proyecto para generar "
        "los artefactos en models/."
    )


def panel(request):
    estimations = Estimation.objects.all()[:10]
    local_projects_count = LocalProject.objects.count()

    context = {
        "estimations_count": Estimation.objects.count(),
        "local_projects_count": local_projects_count,
        "recent_estimations": estimations,
        "model_available": True,
    }
    try:
        metrics = ml.model_metrics()
        mlp = metrics["cv_metrics"]["mlp"]
        cocomo = metrics["cv_metrics"]["cocomo"]
        context["mae_mlp"] = mlp["MAE"]
        context["error_vs_cocomo_pct"] = (mlp["MAE"] - cocomo["MAE"]) / cocomo["MAE"] * 100
        context["error_vs_cocomo_pm"] = mlp["MAE"] - cocomo["MAE"]
        context["cv_metrics"] = metrics["cv_metrics"]
        context["max_mae"] = max(v["MAE"] for v in metrics["cv_metrics"].values())
        context["best_mae_method"] = min(metrics["cv_metrics"], key=lambda m: metrics["cv_metrics"][m]["MAE"])
        context["best_pred25_method"] = max(metrics["cv_metrics"], key=lambda m: metrics["cv_metrics"][m]["PRED25"])
    except FileNotFoundError:
        context["model_available"] = False
        context["model_message"] = _model_unavailable_message()

    return render(request, "estimator/panel.html", context)


def dashboard(request):
    estimations = list(Estimation.objects.order_by("created_at"))
    local_projects = list(LocalProject.objects.all())

    context = {
        "local_projects_count": len(local_projects),
        "high_error_count": sum(1 for p in local_projects if p.error_pct is not None and p.error_pct > 25),
    }

    if estimations:
        context["avg_kloc"] = sum(e.kloc for e in estimations) / len(estimations)
        context["effort_scatter"] = build_effort_scatter(estimations)

        ranking = sorted(estimations, key=lambda e: e.cost_bs, reverse=True)[:5]
        context["ranking"] = ranking
        context["ranking_max"] = ranking[0].cost_bs

        mode_counts = {code: 0 for code, _ in MODE_CHOICES}
        for e in estimations:
            mode_counts[e.mode] += 1
        context["mode_donut"] = build_mode_donut(
            [(label, mode_counts[code], code) for code, label in MODE_CHOICES]
        )
    else:
        context["effort_scatter"] = None
        context["ranking"] = []
        context["mode_donut"] = None

    # Error por metodo en los proyectos locales (MLP y COCOMO sobre el mismo
    # caso) para distinguir si un error alto es del MLP especificamente o de
    # ambos metodos frente al contexto local -- un conteo por balde no lo
    # distingue.
    local_error_rows = []
    pred25_local = None
    for p in local_projects:
        if not p.effort_mlp_pm or not p.effort_real_pm:
            continue
        eaf = ml.eaf_from_ratings(p.cost_driver_values())
        cocomo_pm = ml.cocomo_effort(p.kloc, p.mode, eaf)
        error_mlp = abs(p.effort_real_pm - p.effort_mlp_pm) / p.effort_real_pm * 100
        error_cocomo = abs(p.effort_real_pm - cocomo_pm) / p.effort_real_pm * 100
        local_error_rows.append({"name": p.project_name, "error_mlp": error_mlp, "error_cocomo": error_cocomo})
    context["local_error_bars"] = build_local_error_bars(local_error_rows)
    context["local_error_n"] = len(local_error_rows)
    if local_error_rows:
        pred25_local = sum(1 for r in local_error_rows if r["error_mlp"] <= 25) / len(local_error_rows)
    context["pred25_local"] = pred25_local

    try:
        metrics = ml.model_metrics()
        context["pred25"] = metrics["cv_metrics"]["mlp"]["PRED25"]
        context["pred25_n"] = len(ml.eval_predictions())
    except FileNotFoundError:
        context["pred25"] = None
        context["pred25_n"] = None

    return render(request, "estimator/dashboard.html", context)


def new_estimation(request):
    result = None
    if request.method == "POST":
        form = NewEstimationForm(request.POST)
        if form.is_valid():
            try:
                result = ml.estimate(
                    kloc=form.cleaned_data["kloc"],
                    mode=form.cleaned_data["mode"],
                    ratings=form.cost_driver_values(),
                    rate_per_pm=form.cleaned_data["rate_per_pm"],
                )
            except FileNotFoundError:
                messages.error(request, _model_unavailable_message())
            else:
                if "save" in request.POST:
                    Estimation.objects.create(
                        project_name=form.cleaned_data["project_name"],
                        kloc=form.cleaned_data["kloc"],
                        mode=form.cleaned_data["mode"],
                        rate_per_pm=form.cleaned_data["rate_per_pm"],
                        effort_mlp_pm=result["effort_mlp_pm"],
                        effort_cocomo_pm=result["effort_cocomo_pm"],
                        cost_bs=result["cost_bs"],
                        duration_months=result["duration_months"],
                        staff_count=result["staff_count"],
                        **form.cost_driver_values(),
                    )
                    messages.success(request, "Estimación guardada.")
                    return redirect("estimator:panel")
    else:
        form = NewEstimationForm()

    return render(request, "estimator/new_estimation.html", {
        "form": form,
        "result": result,
        "cost_driver_names": COST_DRIVER_FIELDS,
    })


def model_eval(request):
    context = {"model_available": True}
    try:
        context["metrics"] = ml.model_metrics()
        context["scatter"] = build_scatter(ml.eval_predictions())
    except FileNotFoundError:
        context["model_available"] = False
        context["model_message"] = _model_unavailable_message()
        return render(request, "estimator/model_eval.html", context)

    try:
        context["loss_curve"] = build_loss_curve(ml.train_history())
    except FileNotFoundError:
        context["loss_curve"] = None

    return render(request, "estimator/model_eval.html", context)


def local_projects(request):
    if request.method == "POST":
        form = LocalProjectForm(request.POST)
        if form.is_valid():
            project = form.save(commit=False)
            try:
                eaf = ml.eaf_from_ratings(project.cost_driver_values())
                project.effort_mlp_pm = ml.mlp_effort(project.kloc, project.mode, eaf)
            except FileNotFoundError:
                pass
            project.save()
            messages.success(request, "Proyecto local registrado.")
            return redirect("estimator:local_projects")
    else:
        form = LocalProjectForm()

    projects = LocalProject.objects.all()
    return render(request, "estimator/local_projects.html", {
        "form": form,
        "projects": projects,
    })
