from django.conf.locale.es.formats import *  # noqa: F401,F403

# Convencion boliviana/latinoamericana: punto para miles, coma para
# decimales (la libreria Django trae por defecto espacio para miles en
# locale 'es', que no es lo que se usa en Bolivia).
THOUSAND_SEPARATOR = "."
DECIMAL_SEPARATOR = ","
NUMBER_GROUPING = 3
