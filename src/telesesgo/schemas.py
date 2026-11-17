"""Esquemas de salida estructurada del LLM (validados con Pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Scale = Literal[-2, -1, 0, 1, 2]

Section = Literal[
    "sumario", "politica_nacional", "politica_autonomica", "tribunales_corrupcion", "economia_trabajo",
    "internacional", "sucesos_seguridad", "sociedad", "sanidad", "educacion", "medio_ambiente_clima",
    "inmigracion", "ciencia_tecnologia", "cultura_espectaculos", "deportes", "tiempo", "otros",
]

StoryFormat = Literal[
    "sumario_titulares", "pieza_editada", "directo_conexion", "entrevista", "colas_presentador",
    "opinion_comentario", "promocion_autopromo", "otro",
]

Frame = Literal[
    "conflicto",                    # enfrentamiento entre actores, ganadores/perdedores
    "interes_humano",               # historias personales, emoción
    "consecuencias_economicas",     # costes, impacto en bolsillo/empleo
    "moralidad",                    # valores, juicio ético o religioso
    "atribucion_responsabilidad",   # quién causa o debe resolver el problema
    "estrategia_juego_politico",    # cálculo electoral, encuestas, tácticas
    "seguridad_orden_publico",      # amenaza, delincuencia, control
    "gestion_eficacia",             # buena/mala gestión, competencia técnica
    "informativo_neutro",           # exposición factual sin encuadre dominante
]

Bloc = Literal[
    "gobierno", "partido_gobierno", "socio_gobierno", "oposicion", "gobierno_autonomico",
    "institucion_estado", "justicia", "sindicato", "patronal_empresa", "sociedad_civil",
    "experto", "ciudadano", "internacional", "periodista", "otro",
]


class Story(BaseModel):
    start_line: int = Field(description="Índice de la primera línea de la pieza")
    end_line: int = Field(description="Índice de la última línea de la pieza (inclusive)")
    headline: str = Field(description="Titular neutro y descriptivo, sin adjetivos valorativos, máx. 15 palabras")
    summary: str = Field(description="Resumen factual de 1-2 frases")
    section: Section
    story_format: StoryFormat
    tags: list[str] = Field(description="3-6 etiquetas temáticas en minúscula")
    is_political: bool = Field(description="True si trata de política, instituciones, gobierno o conflicto político/social")


class SpeakerId(BaseModel):
    label: str = Field(description="Etiqueta del hablante tal como aparece en la transcripción, p. ej. 'H03'")
    role: Literal["presentador", "reportero", "fuente", "desconocido"] = Field(
        description="presentador = conductor del informativo; reportero = redactor, enviado o voz en off de la cadena; "
                    "fuente = cualquier persona cuya voz se reproduce (políticos, entrevistados, manifestantes...)")
    name: str | None = Field(description="Nombre completo si el informativo lo dice o es inequívoco; null si no")
    evidence_line: int | None = Field(description="Línea donde se le nombra o se deduce su identidad")


class Segmentation(BaseModel):
    stories: list[Story]
    speakers: list[SpeakerId] = Field(
        default_factory=list,
        description="Identificación de cada etiqueta de hablante (solo si la transcripción trae etiquetas Hnn)")


Voice = Literal["medio", "fuente"]
VOICE_HELP = ("medio = lo dice el propio informativo (presentador, redactor, reportero, voz en off); "
              "fuente = lo dice una persona cuya voz se reproduce (total, declaración, entrevista, manifestante...)")


class Quote(BaseModel):
    quote: str = Field(description="Fragmento LITERAL copiado de la transcripción (5-40 palabras)")
    line: int = Field(description="Índice de la línea donde aparece")
    speaker: str = Field(description="Quién pronuncia el fragmento: nombre, o 'presentador', 'reportero', 'manifestante'...")
    voice: Voice = Field(description=VOICE_HELP)
    note: str = Field(description="Por qué este fragmento apoya la valoración")


class ActorMention(BaseModel):
    name: str = Field(description="Nombre canónico completo, p. ej. 'Pedro Sánchez', 'Partido Popular', 'Tribunal Supremo'")
    kind: Literal["persona", "partido", "gobierno", "institucion", "organizacion", "empresa", "pais", "otro"]
    affiliation: str | None = Field(description="Partido u organización a la que pertenece (null si no aplica)")
    bloc: Bloc
    role: Literal["protagonista", "fuente_con_voz", "citado_indirectamente", "mencionado", "objeto_de_critica"]
    speaks_on_air: bool = Field(description="True si se escucha su propia voz (total/declaración/entrevista)")
    voice_lines: list[int] = Field(description="Índices de líneas donde habla con su propia voz")
    mention_lines: list[int] = Field(description="Índices de líneas donde es mencionado por otros")
    tone: Scale = Field(description="Tono de la pieza hacia este actor: -2 muy negativo .. +2 muy positivo")
    tone_rationale: str


class LoadedTerm(BaseModel):
    quote: str = Field(description="Expresión literal con carga valorativa")
    line: int
    speaker: str = Field(description="Quién la pronuncia: nombre, o 'presentador', 'reportero'...")
    voice: Voice = Field(description=VOICE_HELP + ". Solo cuenta como lenguaje cargado del informativo si es 'medio'")
    connotation: Literal["positiva", "negativa"]
    target: str = Field(description="Actor o asunto al que se aplica")


class Dimension(BaseModel):
    score: Scale | None = Field(description="Puntuación; null si la dimensión no aplica a esta pieza")
    rationale: str
    evidence: list[Quote]


class StoryAnalysis(BaseModel):
    actors: list[ActorMention]
    ideological_lean: Dimension = Field(
        description="Orientación implícita: -2 favorece claramente marcos/actores de izquierda, +2 de derecha, 0 neutral"
    )
    lean_confidence: Literal["baja", "media", "alta"]
    tone_government: Dimension = Field(description="Tono hacia el Gobierno de España y sus miembros: -2 .. +2")
    tone_opposition: Dimension = Field(description="Tono hacia la oposición estatal (PP, Vox...): -2 .. +2")
    primary_frame: Frame
    secondary_frames: list[Frame]
    frame_rationale: str
    frame_evidence: list[Quote]
    source_balance: Literal["no_aplica", "unilateral", "desequilibrado", "equilibrado"] = Field(
        description="Equilibrio de fuentes con voz entre posturas enfrentadas"
    )
    missing_perspectives: list[str] = Field(description="Posturas relevantes que no reciben voz")
    loaded_language: list[LoadedTerm]
    editorializing: bool = Field(description="True si presentador/redactor emite opiniones propias (voz 'medio'); "
                                             "las opiniones de las fuentes NO cuentan")
    editorializing_evidence: list[Quote] = Field(description="Solo citas con voice='medio'")
    selection_notes: str = Field(description="Observaciones sobre qué se omite, el orden o el énfasis")


class EventGroup(BaseModel):
    label: str = Field(description="Descripción neutra del acontecimiento común")
    section: Section
    story_ids: list[int]


class EventMatching(BaseModel):
    events: list[EventGroup]


class Observation(BaseModel):
    lead: str = Field(description="Idea principal en una frase corta (irá en negrita)")
    text: str = Field(description="Desarrollo en 1-3 frases con las cifras y citas del dossier que lo sostienen")


class EditionNotes(BaseModel):
    title: str = Field(description="Titular de la franja: qué la caracteriza en pocas palabras")
    findings: list[Observation] = Field(description="3-5 observaciones sobre cómo cubre cada programa la franja")
    same_story: list[Observation] = Field(description="2-4 noticias comunes contadas de forma distinta; vacío si no hay cruce")
    caveats: list[str] = Field(description="1-3 cautelas sobre los datos de esta franja")
