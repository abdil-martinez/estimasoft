from django.db import models

MODE_CHOICES = [
    ("organic", "Orgánico"),
    ("semidetached", "Semiacoplado"),
    ("embedded", "Empotrado"),
]

RATING_CHOICES = [
    ("vl", "Muy bajo"),
    ("l", "Bajo"),
    ("n", "Nominal"),
    ("h", "Alto"),
    ("vh", "Muy alto"),
    ("xh", "Extra alto"),
]

COST_DRIVER_FIELDS = [
    "rely", "data", "cplx", "time", "stor", "virt", "turn",
    "acap", "aexp", "pcap", "vexp", "lexp", "modp", "tool", "sced",
]

COST_DRIVER_LABELS = {
    "rely": "Fiabilidad requerida",
    "data": "Tamaño de la base de datos",
    "cplx": "Complejidad del producto",
    "time": "Restricción de tiempo de ejecución",
    "stor": "Restricción de almacenamiento",
    "virt": "Volatilidad de la máquina virtual",
    "turn": "Tiempo de respuesta en desarrollo",
    "acap": "Capacidad de analistas",
    "aexp": "Experiencia en la aplicación",
    "pcap": "Capacidad de programadores",
    "vexp": "Experiencia en la máquina virtual",
    "lexp": "Experiencia en el lenguaje",
    "modp": "Uso de prácticas modernas",
    "tool": "Uso de herramientas",
    "sced": "Plazo de desarrollo",
}


class CostDriverMixin(models.Model):
    """Campos comunes a cualquier proyecto evaluado con COCOMO intermedio:
    tamaño, modo de desarrollo y los 15 factores de costo (sec. 9.3 del perfil)."""

    kloc = models.FloatField("Tamaño estimado (KLOC)")
    mode = models.CharField("Modo de desarrollo", max_length=20, choices=MODE_CHOICES)

    rely = models.CharField("Fiabilidad requerida", max_length=2, choices=RATING_CHOICES, default="n")
    data = models.CharField("Tamaño de la base de datos", max_length=2, choices=RATING_CHOICES, default="n")
    cplx = models.CharField("Complejidad del producto", max_length=2, choices=RATING_CHOICES, default="n")
    time = models.CharField("Restricción de tiempo de ejecución", max_length=2, choices=RATING_CHOICES, default="n")
    stor = models.CharField("Restricción de almacenamiento", max_length=2, choices=RATING_CHOICES, default="n")
    virt = models.CharField("Volatilidad de la máquina virtual", max_length=2, choices=RATING_CHOICES, default="n")
    turn = models.CharField("Tiempo de respuesta en desarrollo", max_length=2, choices=RATING_CHOICES, default="n")
    acap = models.CharField("Capacidad de analistas", max_length=2, choices=RATING_CHOICES, default="n")
    aexp = models.CharField("Experiencia en la aplicación", max_length=2, choices=RATING_CHOICES, default="n")
    pcap = models.CharField("Capacidad de programadores", max_length=2, choices=RATING_CHOICES, default="n")
    vexp = models.CharField("Experiencia en la máquina virtual", max_length=2, choices=RATING_CHOICES, default="n")
    lexp = models.CharField("Experiencia en el lenguaje", max_length=2, choices=RATING_CHOICES, default="n")
    modp = models.CharField("Uso de prácticas modernas", max_length=2, choices=RATING_CHOICES, default="n")
    tool = models.CharField("Uso de herramientas", max_length=2, choices=RATING_CHOICES, default="n")
    sced = models.CharField("Plazo de desarrollo", max_length=2, choices=RATING_CHOICES, default="n")

    class Meta:
        abstract = True

    def cost_driver_values(self):
        return {f: getattr(self, f) for f in COST_DRIVER_FIELDS}


class Estimation(CostDriverMixin):
    """Una estimacion registrada desde la pantalla 'Nueva estimacion'."""

    project_name = models.CharField("Nombre del proyecto", max_length=200)
    rate_per_pm = models.FloatField("Tarifa promedio (Bs/persona-mes)")

    effort_mlp_pm = models.FloatField("Esfuerzo estimado - MLP (persona-mes)")
    effort_cocomo_pm = models.FloatField("Esfuerzo estimado - COCOMO intermedio (persona-mes)")
    cost_bs = models.FloatField("Costo estimado (Bs)")
    duration_months = models.FloatField("Duración estimada (meses)")
    staff_count = models.FloatField("Personal estimado")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.project_name


class LocalProject(CostDriverMixin):
    """Proyecto boliviano terminado, con esfuerzo real conocido. Se usa para
    validar el modelo (pantalla 'Proyectos locales') y, a futuro, reentrenarlo."""

    project_name = models.CharField("Nombre del proyecto", max_length=200)
    institution = models.CharField("Institución o empresa", max_length=200, blank=True)

    effort_real_pm = models.FloatField("Esfuerzo real (persona-mes)")
    cost_real_bs = models.FloatField("Costo real (Bs)", null=True, blank=True)
    effort_mlp_pm = models.FloatField("Esfuerzo estimado por el MLP (persona-mes)", null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.project_name

    @property
    def error_pct(self):
        if not self.effort_mlp_pm or not self.effort_real_pm:
            return None
        return abs(self.effort_real_pm - self.effort_mlp_pm) / self.effort_real_pm * 100
