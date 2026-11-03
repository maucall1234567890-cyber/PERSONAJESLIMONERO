# Personaje Limonero

Proyecto de personaje 3D estilizado creado a partir de la referencia `Mascota lima en pose A.png`.

## Contenido

- `Limonero_v6/Limonero_v6.blend` contiene el modelo, la malla continua, el rig FK, controles de ojos, párpados, boca, dedos, muñecas, clavículas y límites de rotación.
- `Limonero_v5/Limonero_v5.blend` contiene el ciclo de reposo con mirada, parpadeo y cierre de boca.
- `Limonero_servir/Limonero_servir.blend` contiene la animación de servir jugo usando únicamente el rig. La escena no añade jarra ni vaso.
- `Limonero_servir/Servir_jugo.mp4` es la previsualización de esa animación.
- Los directorios `sprites` y los atlas PNG contienen las exportaciones para un flujo 2D.

## Controles principales

Abre el archivo en Blender y selecciona `RIG • Limonero` en Pose Mode.

- `Cuerpo["Brazos en descanso"]` mezcla la pose A con la pose de brazos relajados.
- `Clavícula.L` y `Clavícula.R` controlan cada hombro.
- `Brazo`, `Antebrazo`, `Palma` y las dos falanges de cada dedo se pueden posar por separado.
- `Palma["Giro"]` distribuye la pronación o supinación por el antebrazo.
- `Boca["MBP"]`, `A`, `E`, `O` y `U` controlan las formas de habla.
- `Cabeza["Mirada horizontal"]` y `Cabeza["Mirada vertical"]` desplazan la mirada de forma conjunta.
- `Párpado.L["Cerrar"]` y `Párpado.R["Cerrar"]` controlan el parpadeo.

El rig incluye límites de rotación en torso, clavículas, hombros, codos, muñecas, dedos, muslos, rodillas y pies. También incluye correcciones de volumen para hombros, codos, muñecas, torso y rodillas.

## Scripts

- `construir_lima_v4.py` construye la base continua del personaje.
- `pulir_limonero_v6.py` añade soportes, correctivos de volumen, clavículas, límites y la pose de descanso.
- `animar_limonero_v5.py` prepara el reposo, el parpadeo y las formas de boca.
- `servir_jugo.py` crea la animación de servir jugo sin crear objetos de utilería.
- `probar_limonero_v6.py`, `comprobar_rig_v4.py` y `verificar_reposo_v5.py` contienen comprobaciones de malla, pesos, límites, pies y ciclos.

## Flujo de trabajo

1. Se inspeccionó la referencia frontal y se definió una silueta 3D con profundidad estimada.
2. Se generó una malla continua para evitar separaciones entre torso, brazos, manos, piernas y pies.
3. Se creó el esqueleto FK y se normalizaron los pesos de deformación.
4. Se ajustaron dedos, muñecas, hombros, clavículas y transiciones de torso mediante pesos y formas correctivas.
5. Se añadieron controles de boca, ojos, párpados y mirada.
6. Se verificaron poses exigentes, límites de rotación, malla manifold y estabilidad de los pies.
7. Se exportaron previsualizaciones y sprites con fondo transparente.

## Requisitos

- Blender 5.2 LTS o posterior.
- Git LFS es recomendable si se versionan todos los renders y vídeos grandes.

