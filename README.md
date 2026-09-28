# VikoZero — amigo virtual bilingüe para Ollama

VikoZero es un proyecto para entrenar una personalidad conversacional sobre una arquitectura compatible con Ollama. El objetivo no es programación: Viko está orientado a conversación, humor, kaomojis, español e inglés, con conocimientos generales de matemáticas, álgebra, ciencia e historia.

## Flujo automático

1. Publicas este proyecto en GitHub.
2. Abres Google Colab con GPU.
3. Abres `colab_train_viko_zero.py`.
4. Editas **solamente** `REPO_URL`, `VIKO_URL` y `GITHUB_TOKEN`.
5. Ejecutas:

```python
!python colab_train_viko_zero.py
```

El script instala las dependencias, clona el repositorio, toma los datos conversacionales no relacionados con programación, añade el dataset base de Viko, ajusta el modelo, fusiona el adaptador, convierte a GGUF, cuantiza a Q4_K_M, crea `Modelfile` y publica el resultado en el repositorio indicado por `VIKO_URL` usando Git LFS.

## Resultado

El repositorio de salida contiene:

```text
viko-zero.gguf
Modelfile
README.md
```

## Ejecutar con Ollama

En la máquina donde tengas Ollama:

```bash
ollama create viko-zero -f Modelfile
ollama run viko-zero
```

Después de crear el modelo, el comando de chat es simplemente:

```bash
ollama run viko-zero
```

## Perfil de Viko

- Español e inglés.
- Amigo virtual, cálido y conversacional.
- Humor ligero.
- Kaomojis usados de forma natural.
- Matemáticas y álgebra explicadas paso a paso.
- Historia y ciencia en lenguaje sencillo.
- No se entrena como asistente especializado en programación.

## Importante

El entrenamiento usa un modelo base pequeño compatible con Ollama y una afinación LoRA. Esto permite conservar parte del conocimiento general del modelo base mientras se especializa la personalidad de Viko. No es un entrenamiento desde cero.

El GGUF se publica con Git LFS porque es demasiado grande para un archivo Git normal.
