# Benchmark OFF vs ON — RAG Answer

- Fecha: 2026-09-02T08-12-12Z
- Modelo: `qwen/qwen3.5-9b`
- Objetivo: comparar reasoning OFF vs ON en generación RAG.
- Retrieval: idéntico para ambas variantes.
- Temperatura: idéntica dentro de los perfiles.

## Resultados

| Query | Modo | Tiempo (s) | Output | Reasoning | tok/s | TTFT |
|---|---|---:|---:|---:|---:|---:|
| q001 | OFF | 5.401 | 32 | 0 | 43.55 | 0.216 |
| q001 | ON | 81.802 | 3367 | 3350 | 43.51 | 0.372 |
| q002 | OFF | 4.937 | 21 | 0 | 43.46 | 0.179 |
| q002 | ON | 13.541 | 403 | 387 | 43.86 | 0.327 |
| q003 | OFF | 5.135 | 22 | 0 | 43.46 | 0.375 |
| q003 | ON | 14.194 | 434 | 417 | 43.51 | 0.167 |

## Respuestas

### q001: ¿Cómo se conecta el piano al ordenador?

**Chunks recuperados:** knowledge-001, knowledge-000, knowledge-004

#### Reasoning OFF

Según el contexto proporcionado, el dispositivo MIDI (que recibe eventos provenientes del piano digital) se conecta al ordenador mediante una **interfaz USB MIDI**.

#### Reasoning ON

El piano digital se conecta al ordenador mediante una interfaz USB MIDI.

### q002: ¿Dónde se ejecuta la interfaz de usuario?

**Chunks recuperados:** knowledge-002, knowledge-001, knowledge-003

#### Reasoning OFF

Según el contexto proporcionado, la interfaz de usuario se ejecuta en el proceso renderer.

#### Reasoning ON

La interfaz de usuario se ejecuta en el proceso renderer.

### q003: ¿Cuál es el objetivo de la aplicación?

**Chunks recuperados:** knowledge-003, knowledge-004, knowledge-001

#### Reasoning OFF

Según el contexto proporcionado, el objetivo de la aplicación es adaptar progresivamente los ejercicios al usuario.

#### Reasoning ON

El objetivo de la aplicación es adaptar progresivamente los ejercicios al usuario.

## Evaluación

La evaluación de calidad debe considerar principalmente:

1. Corrección respecto del contexto recuperado.
2. Ausencia de información inventada.
3. Claridad y precisión.
4. Coste adicional de reasoning.

## Decisión

### Política adoptada

Para la generación de respuestas RAG factual con `qwen/qwen3.5-9b`:

**Reasoning OFF.**

### Fundamentación

El benchmark utilizó las mismas consultas, el mismo retrieval y el mismo contexto para comparar ambas políticas.

Las tres consultas answerable evaluadas produjeron respuestas correctas y equivalentes en contenido con reasoning OFF y ON.

La diferencia principal estuvo en el coste de inferencia:

- Reasoning OFF: entre 4.937 s y 5.401 s por consulta.
- Reasoning ON: entre 13.541 s y 81.802 s por consulta.
- Reasoning OFF utilizó 0 tokens de reasoning.
- Reasoning ON utilizó entre 387 y 3.350 tokens de reasoning.
- No se observó una mejora cualitativa relevante que justificara ese coste adicional en este workload.

### Decisión arquitectónica

El perfil `RAG_ANSWER_PROFILE` utilizará:

```text
reasoning = off
max_output_tokens = 1024
temperature = 0.2
```

El reasoning ON no se elimina del sistema. Queda disponible mediante `DEEP_ANSWER_PROFILE` para tareas de mayor complejidad.

### Alcance de esta decisión

Esta conclusión es válida para:

- generación factual grounded en contexto recuperado;
    
- consultas similares a las utilizadas en este benchmark;
    
- modelo `qwen/qwen3.5-9b`;
    
- configuración local actual de LM Studio.
    

No debe extrapolarse automáticamente a coding, agentes o tareas de planificación compleja.

Las tareas de desarrollo de software con DeepSeek Harness deberán evaluarse posteriormente mediante un benchmark independiente OFF vs ON.

### Estado

**DECISIÓN: RAG Answer → Reasoning OFF**

**Estado de la Inference Layer: VALIDADA para avanzar a RAGPipeline.**
