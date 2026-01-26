# Reglas Técnicas Infranqueables

> Ninguna regla puede contradecir una decisión congelada.

## 1. Regla de la Soberanía del SQL

**PROHIBIDO:** Lógica de negocio en Frontend. **OBLIGATORIO:** Lógica en DB
(RPCs/Vistas).

## 2. Regla de Commit Inmediato (Safety Net)

**OBLIGATORIO:** Al completar cualquier modificación funcional en el código, se
debe realizar commit y push al repositorio `AudioTexto` inmediatamente.
_Justificación_: Garantizar puntos de restauración seguros ante progresiones
fallidas.

## Meta-Reglas

1. **Regla de Auto-Mejora**: Es obligatorio leer `99_01_Sistema_Aprendizaje.md`
   y realizar un análisis de causa raíz ante cualquier error inesperado.

## Lógica de Negocio y Arquitectura

1. **Protocolo de Inicio**: Está prohibido iniciar cualquier trabajo sin
   verificar este archivo y `99_02_Decisiones_Congeladas.md`.
