# Datos de VikoZero

El entrenador lee archivos `.jsonl` de esta carpeta.

Formatos aceptados:

```json
{"prompt":"hola","response":"Holaa (˶ᵔ ᵕ ᔔ˶)"}
```

o:

```json
{"messages":[{"role":"user","content":"hola"},{"role":"assistant","content":"Holaa (˶ᵔ ᵕ ᔔ˶)"}]}
```

El pipeline filtra automáticamente ejemplos que parezcan centrados en programación para mantener el objetivo de Viko como amigo virtual bilingüe con humor y conocimiento general.
