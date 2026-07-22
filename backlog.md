# Backlog — AI Shorts SaaS

Este documento ordena la evolución del proyecto desde un transcriptor y resumidor hacia un servicio SaaS especializado en convertir vídeos largos en clips cortos editables y listos para publicar.

## Visión del producto

> Pegar una URL o subir un vídeo, recibir varias propuestas de Shorts/Reels, ajustar cortes y subtítulos en un editor sencillo y exportar el resultado en pocos minutos.

El producto no pretende competir inicialmente como editor de vídeo generalista. Su ventaja debe ser la automatización del flujo completo para contenido hablado: descarga, transcripción, detección de momentos, edición asistida, subtítulos y exportación vertical.

### Cliente inicial propuesto

- Creadores de contenido educativo y entrevistas.
- Podcasters y canales de YouTube.
- Coaches, consultores y equipos de marketing pequeños.
- Agencias que producen varios clips por cada vídeo largo.

### Resultado del MVP

Un usuario puede:

1. Crear un proyecto desde una URL de YouTube o un archivo local.
2. Obtener una transcripción con timestamps.
3. Recibir entre 3 y 5 momentos sugeridos por IA.
4. Seleccionar un momento y ajustar sus límites.
5. Corregir y estilizar subtítulos.
6. Elegir formato 9:16, 1:1 o 16:9.
7. Exportar un MP4 H.264 y, opcionalmente, un archivo SRT.

### Métricas que deben guiar el producto

- Tiempo desde importación hasta primera propuesta reproducible.
- Tiempo hasta la primera exportación correcta.
- Porcentaje de propuestas aceptadas o editadas por el usuario.
- Exportaciones completadas por proyecto.
- Coste de procesamiento y almacenamiento por minuto de vídeo.
- Retención semanal de usuarios que completaron una exportación.

## Principios de implementación

- Priorizar recorridos verticales utilizables sobre subsistemas completos aislados.
- Mantener Python, FastAPI, Whisper y FFmpeg como núcleo de procesamiento.
- Representar la edición mediante un documento JSON no destructivo; nunca modificar el original.
- Separar dominio, almacenamiento, trabajos en segundo plano y API para poder migrar de local a SaaS.
- Empezar con una sola pista de vídeo y una pista de subtítulos.
- Medir tiempo, errores y coste antes de añadir facturación.
- Evitar WebGPU, WASM, Rust, escritorio, móvil y plugins hasta validar el flujo principal.

## Flujo de trabajo con GitHub

- Cada elemento implementable debe entregarse en una PR pequeña contra `main`.
- Nombre de rama recomendado: `codex/<id>-<descripcion-corta>`.
- La PR comienza como draft cuando todavía falten criterios de aceptación.
- Una PR no debe mezclar elementos de distintos hitos salvo una dependencia inseparable y documentada.
- Toda PR debe incluir resumen, riesgos, validación ejecutada y capturas cuando cambie la interfaz.
- Los cambios de esquema necesitan migración, prueba de migración y estrategia de reversión.
- Los cambios arquitectónicos relevantes deben actualizar `AGENTS.md`.
- Una funcionalidad incompleta debe quedar detrás de una bandera o sin exposición en producción.

### Estados

- `[ ]` Pendiente
- `[~]` En curso
- `[x]` Completado
- `[!]` Bloqueado

### Prioridades

- **P0**: necesario para el MVP o para evitar pérdida/exposición de datos.
- **P1**: alto valor después de completar el recorrido principal.
- **P2**: mejora posterior; no debe retrasar el MVP.

### Definición de terminado

Un elemento se considera terminado cuando:

- Cumple todos sus criterios de aceptación.
- Tiene pruebas proporcionales al riesgo.
- No introduce secretos, datos personales ni archivos multimedia en Git.
- La documentación de uso o arquitectura está actualizada cuando corresponde.
- La PR contra `main` está aprobada y sus comprobaciones pasan.

## Secuencia recomendada de PRs

| Orden | ID | Prioridad | Entrega | Depende de |
|---:|---|---|---|---|
| 0 | DOC-001 | P0 | Crear y mantener este backlog | — |
| 1 | QLT-001 | P0 | Consolidar pruebas y CI de referencia | DOC-001 |
| 2 | CORE-001 | P0 | Persistir proyectos y recursos multimedia | QLT-001 |
| 3 | CORE-002 | P0 | Persistir y recuperar trabajos de procesamiento | CORE-001 |
| 4 | TRN-001 | P0 | Generar timestamps por palabra | CORE-002 |
| 5 | AI-001 | P0 | Generar propuestas de clips con timestamps | TRN-001 |
| 6 | API-001 | P0 | Exponer proyectos, propuestas y edición por API | AI-001 |
| 7 | WEB-001 | P0 | Crear el shell del nuevo editor web | API-001 |
| 8 | WEB-002 | P0 | Sincronizar vídeo, transcripción y propuestas | WEB-001 |
| 9 | EDIT-001 | P0 | Definir y persistir el documento de edición | WEB-002 |
| 10 | EDIT-002 | P0 | Añadir timeline simple con trim, split y undo | EDIT-001 |
| 11 | CAP-001 | P0 | Editar y previsualizar subtítulos | EDIT-002 |
| 12 | FMT-001 | P0 | Añadir formatos y encuadre | EDIT-002 |
| 13 | EXP-001 | P0 | Renderizar clips con FFmpeg | CAP-001, FMT-001 |
| 14 | EXP-002 | P0 | Ejecutar exportaciones durables y descargables | EXP-001 |
| 15 | MVP-001 | P0 | Validar el recorrido completo del MVP | EXP-002 |
| 16 | AUTH-001 | P0 SaaS | Añadir cuentas y sesiones | MVP-001 |
| 17 | TEN-001 | P0 SaaS | Aislar datos por workspace | AUTH-001 |
| 18 | STOR-001 | P0 SaaS | Migrar medios a almacenamiento de objetos | TEN-001 |
| 19 | USE-001 | P0 SaaS | Medir uso, costes y aplicar cuotas | STOR-001 |
| 20 | OBS-001 | P0 SaaS | Incorporar observabilidad operativa | CORE-002 |
| 21 | PRIV-001 | P0 SaaS | Retención, borrado y controles de privacidad | TEN-001, STOR-001 |
| 22 | BILL-001 | P1 | Añadir planes y facturación | USE-001 |
| 23 | DEP-001 | P0 SaaS | Desplegar staging y producción | OBS-001, PRIV-001 |

## Hito 0 — Base mantenible

### [~] DOC-001 — Backlog de producto y entrega

**Prioridad:** P0

**Resultado:** Existe una fuente única y ordenada para decidir las siguientes PRs.

**Criterios de aceptación:**

- El backlog expresa visión, alcance del MVP y límites explícitos.
- Los elementos están ordenados y tienen identificadores estables.
- El flujo de PRs usa `main` como rama base.

### [ ] QLT-001 — Pruebas y CI de referencia

**Prioridad:** P0

**Resultado:** Los cambios posteriores pueden detectar regresiones del flujo actual.

**Alcance:**

- Corregir pruebas débiles o que dependan del estado global.
- Aislar uploads, resultados, tareas y modelos pesados mediante fixtures.
- Cubrir archivo válido, URL válida, error de descarga, error de transcripción y consulta de estado.
- Ejecutar lint y pruebas en una GitHub Action sobre PRs a `main`.

**Criterios de aceptación:**

- La suite no descarga modelos ni accede a servicios externos.
- Las pruebas se ejecutan de forma repetible en local y CI.
- El workflow bloquea el merge cuando falla una comprobación requerida.

### [ ] CORE-001 — Modelo persistente de proyectos y medios

**Prioridad:** P0

**Resultado:** Un reinicio no elimina proyectos ni su relación con archivos y transcripciones.

**Alcance:**

- Definir `Project`, `MediaAsset`, `Transcript` y sus estados.
- Añadir un repositorio desacoplado con SQLite como implementación local inicial.
- Crear migraciones versionadas.
- Mantener los binarios fuera de la base de datos.

**Criterios de aceptación:**

- Crear, consultar, listar y eliminar proyectos mediante pruebas.
- Un proyecto conserva sus metadatos después de reiniciar la aplicación.
- Las rutas almacenadas no permiten escapar del directorio de medios configurado.

### [ ] CORE-002 — Trabajos durables de procesamiento

**Prioridad:** P0

**Resultado:** Descargas, transcripciones y análisis dejan de depender de un diccionario en memoria.

**Alcance:**

- Definir `ProcessingJob`, estados, progreso, error y timestamps.
- Unificar el procesamiento de URL y archivo en un servicio común.
- Recuperar o marcar de forma segura trabajos interrumpidos tras un reinicio.
- Diseñar una interfaz de cola que permita una implementación Redis en producción.

**Criterios de aceptación:**

- El estado de un trabajo persiste tras reiniciar el proceso web.
- Los fallos conservan un mensaje seguro para el usuario y detalle técnico en logs.
- Reintentar no crea resultados duplicados ni corrompe el proyecto.

## Hito 1 — Inteligencia aplicada al clipping

### [ ] TRN-001 — Timestamps por palabra y captions normalizados

**Prioridad:** P0

**Resultado:** La transcripción puede alimentar subtítulos precisos y selección de clips.

**Alcance:**

- Activar timestamps por palabra en Faster Whisper.
- Normalizar palabras y segmentos en modelos tipados.
- Mantener compatibilidad con resultados antiguos que solo tengan segmentos.
- Exportar SRT básico desde la transcripción.

**Criterios de aceptación:**

- Cada palabra válida incluye inicio, final y texto.
- Los timestamps son crecientes y permanecen dentro de la duración del medio.
- Existe una prueba con fixture corto que valida segmentación y SRT.

### [ ] AI-001 — Generador de propuestas de clips

**Prioridad:** P0

**Resultado:** Un proyecto produce entre 3 y 5 candidatos reproducibles, no solo un resumen textual.

**Modelo mínimo de propuesta:**

- `start`, `end` y `duration`.
- `title`, `hook`, `reason` y `score`.
- Segmentos de transcripción utilizados.
- Estado de aceptación, descarte o edición.

**Criterios de aceptación:**

- Las propuestas respetan una duración configurable y no cortan palabras.
- El resultado se valida contra un esquema antes de persistirse.
- Existe un fallback determinista cuando el proveedor de IA no responde.
- El prompt y la respuesta no escriben la transcripción completa en logs.

### [ ] API-001 — API de proyectos, propuestas y selección

**Prioridad:** P0

**Resultado:** El frontend puede operar el flujo completo sin acceder al sistema de archivos.

**Alcance:**

- Endpoints para crear/listar/consultar proyectos.
- Endpoint para consultar progreso mediante polling; eventos en tiempo real quedan para después.
- Endpoints para listar, aceptar, descartar y regenerar propuestas.
- Respuestas tipadas y errores consistentes.

**Criterios de aceptación:**

- OpenAPI refleja los modelos reales.
- La API nunca expone rutas absolutas del servidor.
- Las pruebas cubren estados válidos, recursos inexistentes y transiciones inválidas.

## Hito 2 — Editor enfocado

### [ ] WEB-001 — Shell React y TypeScript del editor

**Prioridad:** P0

**Resultado:** Existe una base mantenible para el estado complejo del editor sin eliminar prematuramente el flujo web actual.

**Alcance:**

- Elegir y documentar el sistema de build.
- Crear rutas de proyectos e editor.
- Integrar el artefacto frontend con FastAPI/Docker mediante build reproducible.
- Añadir manejo de carga, error y proyecto inexistente.

**Criterios de aceptación:**

- El build frontend es reproducible en CI y Docker.
- La aplicación actual sigue permitiendo importar un medio.
- La ruta del editor abre un proyecto real desde la API.

### [ ] WEB-002 — Reproductor, transcripción y propuestas sincronizadas

**Prioridad:** P0

**Resultado:** El usuario puede evaluar rápidamente cada propuesta.

**Alcance:**

- Reproductor con seek desde un segmento o palabra.
- Lista de propuestas con preview, duración, score y motivo.
- Acciones aceptar, descartar y abrir en editor.
- Resaltado del texto correspondiente al tiempo actual.

**Criterios de aceptación:**

- Seleccionar una propuesta reproduce exactamente su intervalo.
- La reproducción se detiene o reinicia al alcanzar el final del candidato.
- Controles esenciales son utilizables con teclado.

### [ ] EDIT-001 — Documento de edición no destructiva

**Prioridad:** P0

**Resultado:** Frontend y backend comparten una representación versionada de la edición.

**Modelo inicial:**

- Fuente multimedia y versión del documento.
- Clips con `sourceStart`, `sourceEnd` y posición en timeline.
- Aspect ratio, resolución, crop y fondo.
- Captions y preset visual.
- Opciones de audio y exportación.

**Criterios de aceptación:**

- El esquema rechaza intervalos imposibles o solapamientos no soportados.
- Guardar y volver a abrir conserva la edición.
- Las versiones antiguas tienen una estrategia explícita de migración.

### [ ] EDIT-002 — Timeline de una pista

**Prioridad:** P0

**Resultado:** El usuario puede ajustar el contenido sugerido sin necesitar un editor profesional.

**Alcance:**

- Playhead, zoom y waveform simplificado.
- Trim de inicio y final.
- Split en el playhead, eliminar y reordenar clips.
- Undo/redo mediante comandos.

**Criterios de aceptación:**

- Ninguna operación modifica el archivo fuente.
- Undo/redo restaura exactamente el documento anterior.
- Los límites respetan duración mínima y duración de la fuente.
- Timeline y reproductor permanecen sincronizados.

### [ ] CAP-001 — Editor y presets de subtítulos

**Prioridad:** P0

**Resultado:** El usuario puede corregir el texto y obtener subtítulos legibles en vídeo vertical.

**Alcance:**

- Edición de texto y tiempos.
- División y unión de captions.
- Presets iniciales: limpio, alto contraste y palabra activa.
- Posición, tamaño y colores con límites seguros.

**Criterios de aceptación:**

- Cambios de texto y tiempo aparecen en preview y persisten.
- El sistema evita captions vacíos o tiempos invertidos.
- El mismo documento produce SRT y subtítulos quemados coherentes.

### [ ] FMT-001 — Formatos y encuadre

**Prioridad:** P0

**Resultado:** Un clip puede prepararse para Shorts/Reels, cuadrado o landscape.

**Alcance:**

- Presets 9:16, 1:1 y 16:9.
- Crop manual con preview.
- Modos fit, fill y fondo desenfocado.
- Resoluciones de salida seguras y configurables.

**Criterios de aceptación:**

- Cambiar formato no altera los tiempos de edición.
- El crop se valida dentro de los límites de la fuente.
- Preview y render utilizan la misma interpretación del encuadre.

## Hito 3 — Exportación confiable

### [ ] EXP-001 — Motor de render FFmpeg

**Prioridad:** P0

**Resultado:** El documento de edición genera un MP4 reproducible.

**Alcance:**

- Traducir el documento de edición a una invocación FFmpeg segura.
- Cortar y concatenar clips.
- Aplicar crop, scale, fondo, audio y captions.
- Exportar MP4 H.264/AAC con presets de calidad limitados.

**Criterios de aceptación:**

- No se construyen comandos mediante interpolación de shell insegura.
- La duración del resultado coincide con la timeline dentro de una tolerancia documentada.
- Audio y vídeo permanecen sincronizados.
- Fixtures de 9:16 y 16:9 se validan con `ffprobe`.

### [ ] EXP-002 — Trabajos de exportación y descargas

**Prioridad:** P0

**Resultado:** Exportar no bloquea el servidor web y el usuario conoce el progreso.

**Alcance:**

- Cola, cancelación y reintentos controlados.
- Progreso persistente y estados terminales claros.
- URL de descarga temporal o endpoint autorizado.
- Política de limpieza de temporales y exportaciones caducadas.

**Criterios de aceptación:**

- Reiniciar el servidor web no pierde el registro de la exportación.
- Dos solicitudes idénticas no pisan el mismo archivo.
- Cancelar termina el proceso hijo y limpia temporales.

### [ ] MVP-001 — Recorrido end-to-end y beta local

**Prioridad:** P0

**Resultado:** El flujo prometido por el MVP funciona de principio a fin en Docker.

**Criterios de aceptación:**

- Una prueba end-to-end usa un fixture propio y no accede a YouTube.
- URL/archivo → transcripción → propuesta → edición → MP4 funciona en un entorno limpio.
- Los errores de cada etapa se muestran con una acción de recuperación.
- README y documentación de Docker explican el flujo completo.

## Hito 4 — Preparación SaaS

Este hito comienza después de validar que usuarios reales completan exportaciones. Las decisiones de proveedor deben quedar detrás de interfaces para evitar acoplamiento innecesario.

### [ ] AUTH-001 — Cuentas y sesiones

**Prioridad:** P0 SaaS

**Criterios de aceptación:**

- Registro/inicio de sesión o proveedor externo documentado.
- Cookies seguras, protección CSRF cuando corresponda y cierre de sesión.
- Los endpoints privados rechazan sesiones inexistentes o expiradas.

### [ ] TEN-001 — Workspaces y aislamiento multi-tenant

**Prioridad:** P0 SaaS

**Criterios de aceptación:**

- Todo proyecto, medio, trabajo y exportación pertenece a un workspace.
- Consultas y descargas verifican pertenencia en el servidor.
- Pruebas negativas demuestran que un usuario no accede a datos de otro workspace.

### [ ] STOR-001 — Almacenamiento de objetos

**Prioridad:** P0 SaaS

**Criterios de aceptación:**

- Existe una interfaz común para filesystem local y almacenamiento S3-compatible.
- Uploads y descargas grandes no atraviesan innecesariamente la memoria del proceso web.
- Los objetos usan claves no predecibles y URLs temporales.
- El borrado de proyecto elimina o agenda todos sus objetos asociados.

### [ ] USE-001 — Medición de uso y cuotas

**Prioridad:** P0 SaaS

**Medidas mínimas:**

- Minutos importados y transcritos.
- Minutos exportados.
- Bytes almacenados.
- Uso de proveedor de IA y tiempo de worker.

**Criterios de aceptación:**

- Los eventos son idempotentes y auditables.
- Los límites se aplican en el servidor antes de comenzar trabajo costoso.
- El usuario puede consultar su consumo y entender por qué un trabajo fue rechazado.

### [ ] OBS-001 — Logs, métricas y trazabilidad

**Prioridad:** P0 SaaS

**Criterios de aceptación:**

- Cada petición y trabajo tiene un identificador de correlación.
- Logs estructurados no contienen tokens, cookies, transcripciones completas ni URLs firmadas.
- Se miden duración, cola, éxito y error por etapa.
- Existen alertas para acumulación de trabajos y tasa anormal de fallos.

### [ ] PRIV-001 — Privacidad, retención y borrado

**Prioridad:** P0 SaaS

**Criterios de aceptación:**

- La política de retención de originales, temporales y exportaciones es configurable.
- El usuario puede borrar un proyecto y solicitar borrado de cuenta.
- El proceso de borrado abarca base de datos, objetos, cachés y trabajos pendientes.
- Antes de la beta pública se revisan derechos de contenido y términos de las plataformas importadas.

### [ ] BILL-001 — Planes y facturación

**Prioridad:** P1

**Criterios de aceptación:**

- Los planes se expresan como permisos y cuotas internas, no como condicionales dispersos.
- Los webhooks son verificados, idempotentes y reintentables.
- Cancelaciones, impagos y cambios de plan tienen estados probados.
- El sistema puede operar en modo gratuito limitado sin proveedor de pago durante desarrollo.

### [ ] DEP-001 — Staging y producción

**Prioridad:** P0 SaaS

**Criterios de aceptación:**

- Web, worker, base de datos y almacenamiento tienen configuración separada por entorno.
- Migraciones se ejecutan de forma controlada.
- Secretos no viven en imágenes Docker ni en el repositorio.
- Existen health checks, backups verificados y procedimiento de rollback.

## Hito 5 — Crecimiento, después del MVP

### [ ] GROW-001 — Plantillas reutilizables

**Prioridad:** P1

- Guardar estilos de captions, formato, crop y exportación como plantilla.
- Aplicar una plantilla sin alterar la fuente ni los timestamps.

### [ ] GROW-002 — Exportación de variantes por lote

**Prioridad:** P1

- Generar múltiples propuestas y formatos desde un proyecto.
- Limitar concurrencia y mostrar coste estimado antes de iniciar.

### [ ] GROW-003 — Brand kits y equipos

**Prioridad:** P1

- Colores, fuentes, logos y presets por workspace.
- Roles mínimos de propietario y miembro.

### [ ] GROW-004 — Música, imágenes y B-roll

**Prioridad:** P2

- Añadir una pista secundaria limitada.
- Controlar volumen, fades y atribución/licencia de recursos externos.

### [ ] GROW-005 — Publicación asistida

**Prioridad:** P2

- Generar título, descripción, hashtags y thumbnail.
- Mantener la publicación directa fuera de alcance hasta validar permisos y APIs por plataforma.

## Fuera de alcance por ahora

- Editor multipista profesional completo.
- Colaboración simultánea en tiempo real.
- Keyframes avanzados, máscaras, tracking y efectos GPU.
- Compositor WebGPU/WASM o núcleo Rust.
- Aplicaciones nativas de escritorio y móvil.
- Marketplace de plugins y servidor MCP.
- Clonar toda la experiencia o arquitectura de OpenCut.

## Riesgos que deben revisarse en cada hito

- **Coste:** transcripción, IA, render y almacenamiento pueden destruir el margen si no se miden.
- **Fiabilidad:** los trabajos largos deben sobrevivir reinicios y admitir reintentos seguros.
- **Seguridad:** uploads, URLs, FFmpeg y separación entre tenants amplían la superficie de ataque.
- **Privacidad:** vídeos y transcripciones pueden contener datos sensibles.
- **Derechos de contenido:** el usuario debe tener autorización para descargar, procesar y publicar el material.
- **Alcance:** las funciones de editor generalista pueden retrasar indefinidamente la propuesta principal.

## Preguntas de producto pendientes

- ¿Cuál será el primer nicho: podcasts, educación, coaches o agencias?
- ¿Qué duración por defecto deben tener los clips sugeridos?
- ¿El primer despliegue usará CPU, GPU dedicada o proveedor de transcripción?
- ¿Cuánto tiempo se conservarán originales y exportaciones por plan?
- ¿La primera beta será gratuita con invitación o tendrá pago desde el inicio?
- ¿Qué idiomas deben recibir calidad de primera clase en el MVP?
