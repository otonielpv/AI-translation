# Traducción en el PC de streaming

El perfil de `config.yaml` está pensado como punto de partida para el i9,
32 GB de RAM y RTX 3050 de la iglesia. La memoria de vídeo aún debe confirmarse.
No se han medido tiempos reales con ese equipo ni se ha validado la calidad
con grabaciones de la iglesia.

El objetivo de aceptación es escuchar la traducción con unos segundos de
retraso, sin alcanzar de forma habitual 10–15 s ni acumular demora durante el
culto. El perfil equilibrado da más contexto y alternativas que el perfil
mínimo de latencia. No garantiza un tiempo concreto sin medir el equipo.

## Cambios

| Ajuste | Antes | Ahora |
| --- | --- | --- |
| Segmento máximo de voz | 8 s | 4 s (aproximado, pasos de 32 ms) |
| Silencio para cerrar frase | 700 ms | 500 ms |
| Whisper | medium, beam 5 | small, beam 3, int8_float16 en CUDA |
| Traducción | búsqueda por defecto del modelo | beam 2, CPU, modo inferencia |
| Hilos | valores por defecto | 2 para PyTorch y 2 para Whisper |
| Colas de procesamiento | 50 fragmentos por etapa | 3, conservando los recientes |
| Cola de difusión / por oyente | 200 / 20 fragmentos | 3 / 3 |
| Silencio entre voces | 200 ms | 50 ms |
| WAV de depuración | escritura continua | desactivados |

El VAD se carga antes de capturar audio. Whisper comprueba CUDA con CTranslate2,
el motor que realmente ejecuta la transcripción, independientemente de si
PyTorch se instaló con CUDA. Si la carga CUDA falla, el log muestra el motivo
y se intenta CPU con int8. Eso puede aumentar mucho la latencia.

La traducción y Piper siguen en CPU para reservar GPU para Whisper y streaming.
El límite de hilos de PyTorch no limita todo el proceso: Piper/ONNX y otras
bibliotecas pueden usar sus propios hilos.

## Qué ocurre con la sobrecarga

Al llenarse una cola se omite el fragmento pendiente más antiguo y se registra
en `logs/app.log`. `performance.max_segment_age_seconds: 7` descarta voz
demasiado antigua antes de STT, traducción y TTS, y otra vez después de TTS.
Esto evita seguir gastando recursos en una cola atrasada; **puede omitir contenido**.
Este presupuesto deja margen para red y reproducción. No es una garantía de
retraso máximo hasta los auriculares: las omisiones son una recuperación de
emergencia, no una mejora de velocidad de los modelos.

El navegador decodifica en orden, limita a 3 los fragmentos esperando
decodificación y acelera a 1,12× cuando hay más de 2 s programados por delante.
Si hay más de 2 s de espera programada, o el audio pendiente más el nuevo
fragmento supera 8 s, elimina las voces
futuras y conserva la que ya está sonando. Un fragmento individual largo puede
superar ese umbral. El oyente ve un aviso cuando se omiten fragmentos.
Al desconectar se cancela también el audio ya programado.

Los segmentos de 4 s y las búsquedas beam 3/2 permiten más contexto y alternativas
que el perfil anterior de 3 s y beam 1/1; su efecto en precisión debe comprobarse.
Comprobar especialmente nombres bíblicos, citas y frases largas en alemán.
Si la precisión no basta y hay margen medido de GPU y tiempo, probar Whisper
`medium` como único cambio, comparando calidad y retraso bajo carga. Si el
servidor tarda cerca de 7 s, ya no hay margen para aumentar su carga. No elevar
el umbral de descarte para ocultar ese problema.

## Prueba en la iglesia

1. Instalar Piper en el mismo entorno con el que se ejecuta `run.bat`:
   `python -m pip install piper-tts`. El fallback por ejecutable recarga la voz
   con cada fragmento y muestra una advertencia. Descargar los modelos antes
   del culto; el primer inicio incluye carga/descarga y no sirve de benchmark.
2. Ejecutar con normalidad el streaming y las diapositivas. Confirmar en el log
   que Whisper carga `small` en `cuda (int8_float16)` sin fallback a CPU.
   Las bibliotecas CUDA/cuDNN que necesita CTranslate2 deben estar disponibles;
   instalar PyTorch con CUDA por sí solo no garantiza esto.
3. Probar 15–20 minutos de una misma predicación desde la entrada habitual, o
   `python main.py --play-file "C:\\grabaciones\\predicacion.wav"` (requiere ffmpeg
   y la selección de un dispositivo de entrada válido en la versión actual).
4. Anotar el tiempo desde una frase reconocible hasta oírla en un móvil al
   principio y al final. Revisar también fluidez del streaming, uso de GPU/VRAM
   y CPU en el Administrador de tareas, y calidad del alemán.
5. Revisar las líneas `[latency]`, `overloaded`, `Skipping speech older` y
   `STT slower than real time` de `logs/app.log`. Omisiones recurrentes indican
   que el perfil aún no sostiene esa carga, aunque parezca recuperar el directo.
   La prueba solo es satisfactoria si mantiene precisión aceptable, retraso
   inferior a 10 s en el móvil y ausencia de omisiones recurrentes. Medir desde
   el inicio de la frase original hasta el inicio de su voz traducida.

`~server ready` del operador / `user_delay_s` de `/status` estima el tiempo desde
el inicio del segmento hasta terminar TTS, incluyendo su espera en las colas de
procesamiento. Conservamos el nombre JSON por compatibilidad. Es una estimación:
el timestamp se reconstruye al detectar voz, puede omitir espera previa en la
cola de captura y no incluye red, decodificación ni reproducción del móvil.
`pipeline_s` suma únicamente los tiempos de STT, traducción y TTS.

## Verificación automática

```text
python -m unittest discover -s tests -v
node --test tests/listener.test.cjs
python -m compileall -q src main.py tests
```

Las pruebas simulan detección CUDA, segmentación, saturación de colas y audio
del navegador. No descargan modelos ni demuestran rendimiento o calidad real.

Referencias de las opciones usadas:
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[detección de GPU de CTranslate2](https://opennmt.net/CTranslate2/python/ctranslate2.html),
[generación de Transformers](https://huggingface.co/docs/transformers/main_classes/text_generation).
