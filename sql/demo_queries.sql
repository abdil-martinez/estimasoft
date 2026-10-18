-- Consultas de ejemplo sobre la base de datos de EstimaSoft (Postgres,
-- db "estimasoft"), de mas simple a mas compleja. Pensadas para pegar
-- directo en el editor SQL de DBeaver, conectado a:
--   host localhost · puerto 5433 · db/user/pass "estimasoft"
--
-- Corren sobre los datos de ejemplo cargados con scripts/seed_demo_data.py
-- (7 filas en estimator_estimation, 5 en estimator_localproject). Los
-- "esfuerzo real" de estimator_localproject son valores inventados solo para
-- tener datos de prueba -- el modelo real todavia no fue calibrado con
-- proyectos bolivianos reales, asi que el error que muestran las consultas
-- 5 y 9 no refleja el desempeno real del MLP.


-- 1. SELECT simple -----------------------------------------------------
SELECT project_name, institution, kloc, effort_real_pm
FROM estimator_localproject;


-- 2. WHERE + ORDER BY ---------------------------------------------------
SELECT project_name, kloc, mode, effort_mlp_pm, cost_bs
FROM estimator_estimation
WHERE mode = 'semidetached'
ORDER BY effort_mlp_pm DESC;


-- 3. Funciones agregadas (COUNT, AVG, MIN, MAX) --------------------------
SELECT
  COUNT(*) AS total_proyectos,
  ROUND(AVG(effort_mlp_pm)::numeric, 1) AS esfuerzo_promedio_pm,
  ROUND(MIN(effort_mlp_pm)::numeric, 1) AS esfuerzo_minimo,
  ROUND(MAX(effort_mlp_pm)::numeric, 1) AS esfuerzo_maximo
FROM estimator_estimation;


-- 4. GROUP BY -------------------------------------------------------------
SELECT
  mode AS modo_desarrollo,
  COUNT(*) AS cantidad,
  ROUND(AVG(effort_mlp_pm)::numeric, 1) AS esfuerzo_promedio_pm,
  ROUND(AVG(cost_bs)::numeric, 0) AS costo_promedio_bs
FROM estimator_estimation
GROUP BY mode
ORDER BY esfuerzo_promedio_pm DESC;


-- 5. Columna calculada con CASE WHEN ---------------------------------------
SELECT
  project_name,
  institution,
  effort_real_pm,
  ROUND(effort_mlp_pm::numeric, 1) AS effort_mlp_pm,
  ROUND((ABS(effort_real_pm - effort_mlp_pm) / effort_real_pm * 100)::numeric, 1) AS error_pct,
  CASE
    WHEN ABS(effort_real_pm - effort_mlp_pm) / effort_real_pm <= 0.25 THEN 'dentro de PRED(25)'
    ELSE 'fuera de PRED(25)'
  END AS evaluacion
FROM estimator_localproject
ORDER BY error_pct;


-- 6. Subconsulta ------------------------------------------------------------
SELECT project_name, mode, ROUND(effort_mlp_pm::numeric, 1) AS effort_mlp_pm
FROM estimator_estimation
WHERE effort_mlp_pm > (SELECT AVG(effort_mlp_pm) FROM estimator_estimation)
ORDER BY effort_mlp_pm DESC;


-- 7. UNION entre las dos tablas -----------------------------------------------
SELECT project_name, 'estimacion' AS origen, kloc, ROUND(effort_mlp_pm::numeric, 1) AS effort_mlp_pm
FROM estimator_estimation
UNION ALL
SELECT project_name, 'proyecto_local' AS origen, kloc, ROUND(effort_mlp_pm::numeric, 1) AS effort_mlp_pm
FROM estimator_localproject
ORDER BY effort_mlp_pm DESC
LIMIT 6;


-- 8. Funcion de ventana: RANK + PARTITION BY -----------------------------------
SELECT
  project_name,
  mode,
  ROUND(effort_mlp_pm::numeric, 1) AS effort_mlp_pm,
  RANK() OVER (PARTITION BY mode ORDER BY effort_mlp_pm DESC) AS ranking_en_su_modo,
  ROUND((effort_mlp_pm / MAX(effort_mlp_pm) OVER (PARTITION BY mode) * 100)::numeric, 1) AS pct_del_maximo_del_modo
FROM estimator_estimation
ORDER BY mode, ranking_en_su_modo;


-- 9. CTE (WITH): desglosar el calculo en pasos -----------------------------------
WITH errores AS (
  SELECT
    project_name,
    institution,
    effort_real_pm,
    effort_mlp_pm,
    ABS(effort_real_pm - effort_mlp_pm) / effort_real_pm AS error_relativo
  FROM estimator_localproject
),
promedio AS (
  SELECT AVG(error_relativo) AS error_promedio FROM errores
)
SELECT
  e.project_name,
  e.institution,
  ROUND((e.error_relativo * 100)::numeric, 1) AS error_pct,
  ROUND((p.error_promedio * 100)::numeric, 1) AS error_promedio_pct
FROM errores e, promedio p
WHERE e.error_relativo > p.error_promedio
ORDER BY e.error_relativo DESC;


-- 10 y 11. JOIN real (requieren la tabla demo_institution, que se borro) -----
-- Si quieres repetir estas dos, primero recrea la tabla y los datos:
--
-- CREATE TABLE demo_institution (
--   id SERIAL PRIMARY KEY,
--   name VARCHAR(200) UNIQUE NOT NULL,
--   city VARCHAR(100) NOT NULL,
--   sector VARCHAR(50) NOT NULL
-- );
-- INSERT INTO demo_institution (name, city, sector) VALUES
--   ('Alcaldía de El Alto', 'El Alto', 'público'),
--   ('Cooperativa La Paz Ltda.', 'La Paz', 'privado'),
--   ('Distribuidora Andina', 'La Paz', 'privado'),
--   ('Universidad Privada de Santa Cruz', 'Santa Cruz', 'educación'),
--   ('Comercial Cochabamba S.R.L.', 'Cochabamba', 'privado');

-- 10. JOIN simple
-- SELECT lp.project_name, lp.institution, i.city, i.sector,
--        lp.effort_real_pm, ROUND(lp.cost_real_bs::numeric, 0) AS costo_real_bs
-- FROM estimator_localproject lp
-- JOIN demo_institution i ON i.name = lp.institution
-- ORDER BY i.sector, i.city;

-- 11. JOIN + GROUP BY + agregacion
-- SELECT i.sector, COUNT(*) AS cantidad_proyectos,
--        ROUND(AVG(lp.effort_real_pm)::numeric, 1) AS esfuerzo_real_promedio,
--        ROUND(SUM(lp.cost_real_bs)::numeric, 0) AS costo_total_bs
-- FROM estimator_localproject lp
-- JOIN demo_institution i ON i.name = lp.institution
-- GROUP BY i.sector
-- ORDER BY costo_total_bs DESC;
