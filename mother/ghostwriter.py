"""
Ghostwriting — voice-matched text generation.

LEAF module (stdlib only). Extracts a voice signature from user messages
and generates LLM instruction blocks for writing in the user's voice.

Genome #135: ghostwriting-capable — writes in the user's voice.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class VoiceSignature:
    """Extracted writing style profile from user messages."""

    avg_sentence_length: float          # words per sentence
    avg_word_length: float              # characters per word
    contraction_rate: float             # 0.0-1.0
    question_rate: float                # 0.0-1.0
    exclamation_rate: float             # 0.0-1.0
    formality: str                      # "formal", "neutral", "casual"
    vocabulary_richness: float          # unique/total word ratio
    sentence_count: int                 # total analyzed
    message_count: int                  # messages analyzed
    frequent_phrases: List[str] = field(default_factory=list)  # recurring 2-grams


@dataclass(frozen=True)
class VoicePersona:
    """LLM instruction block for ghostwriting in a specific voice."""

    instruction: str                    # system prompt injection
    style_notes: List[str]             # bullet-point style guide
    avoid: List[str]                   # things to avoid
    examples: List[str]               # characteristic phrases


def extract_voice_signature(messages: List[str]) -> VoiceSignature:
    """Analyze user messages to extract a voice signature.

    Expects raw message strings (not role-prefixed).
    Needs at least 3 messages for meaningful analysis.
    """
    if not messages:
        return _empty_signature()

    all_sentences: List[str] = []
    all_words: List[str] = []
    contraction_count = 0
    question_count = 0
    exclamation_count = 0
    total_sentences = 0
    bigrams: dict = {}

    for msg in messages:
        sentences = _split_sentences(msg)
        for sent in sentences:
            total_sentences += 1
            words = sent.split()
            all_words.extend(words)
            all_sentences.append(sent)

            if sent.rstrip().endswith("?"):
                question_count += 1
            if sent.rstrip().endswith("!"):
                exclamation_count += 1

            # Count contractions
            contraction_count += len(re.findall(
                r"\b\w+(?:'t|'re|'ve|'ll|'d|'m|'s)\b", sent, re.IGNORECASE
            ))

            # Bigrams
            lower_words = [w.lower().strip(".,!?;:\"'()") for w in words]
            lower_words = [w for w in lower_words if w]
            for i in range(len(lower_words) - 1):
                bg = f"{lower_words[i]} {lower_words[i+1]}"
                bigrams[bg] = bigrams.get(bg, 0) + 1

    if not all_words or total_sentences == 0:
        return _empty_signature()

    # Compute metrics
    avg_sent_len = len(all_words) / total_sentences
    avg_word_len = sum(len(w) for w in all_words) / len(all_words)
    contraction_rate = contraction_count / total_sentences
    question_rate = question_count / total_sentences
    exclamation_rate = exclamation_count / total_sentences

    # Vocabulary richness
    unique = len(set(w.lower() for w in all_words))
    richness = unique / len(all_words)

    # Formality
    formality = _classify_formality(
        contraction_rate, avg_sent_len, exclamation_rate,
    )

    # Frequent phrases (bigrams appearing 3+ times)
    frequent = sorted(
        [(phrase, count) for phrase, count in bigrams.items() if count >= 2],
        key=lambda x: -x[1],
    )[:10]
    frequent_phrases = [phrase for phrase, _ in frequent]

    return VoiceSignature(
        avg_sentence_length=round(avg_sent_len, 1),
        avg_word_length=round(avg_word_len, 1),
        contraction_rate=round(min(contraction_rate, 1.0), 2),
        question_rate=round(question_rate, 2),
        exclamation_rate=round(exclamation_rate, 2),
        formality=formality,
        vocabulary_richness=round(richness, 2),
        sentence_count=total_sentences,
        message_count=len(messages),
        frequent_phrases=frequent_phrases,
    )


def generate_voice_persona(sig: VoiceSignature) -> VoicePersona:
    """Generate an LLM instruction block from a voice signature.

    Produces a system prompt injection that instructs the LLM
    to write in the user's detected style.
    """
    style_notes: List[str] = []
    avoid: List[str] = []

    # Sentence length
    if sig.avg_sentence_length < 10:
        style_notes.append("Use short, punchy sentences (under 10 words average)")
        avoid.append("Long, complex sentence structures")
    elif sig.avg_sentence_length > 20:
        style_notes.append("Use longer, detailed sentences (20+ words average)")
        avoid.append("Overly terse responses")
    else:
        style_notes.append("Use medium-length sentences (10-20 words)")

    # Contractions
    if sig.contraction_rate > 0.3:
        style_notes.append("Use contractions freely (don't, can't, won't)")
        avoid.append("Formal un-contracted forms")
    elif sig.contraction_rate < 0.05:
        style_notes.append("Avoid contractions — write formally")
        avoid.append("Contractions (don't → do not)")

    # Questions
    if sig.question_rate > 0.3:
        style_notes.append("Ask questions frequently — rhetorical and direct")
    elif sig.question_rate < 0.05:
        style_notes.append("Make statements rather than asking questions")

    # Exclamations
    if sig.exclamation_rate > 0.2:
        style_notes.append("Use exclamation marks for emphasis")
    elif sig.exclamation_rate < 0.02:
        avoid.append("Exclamation marks")

    # Formality
    if sig.formality == "casual":
        style_notes.append("Casual, conversational tone")
        avoid.append("Corporate or academic language")
    elif sig.formality == "formal":
        style_notes.append("Professional, measured tone")
        avoid.append("Slang, casual abbreviations")
    else:
        style_notes.append("Balanced, natural tone — neither too formal nor too casual")

    # Vocabulary
    if sig.vocabulary_richness > 0.7:
        style_notes.append("Varied vocabulary — avoid repeating words")
    elif sig.vocabulary_richness < 0.4:
        style_notes.append("Direct vocabulary — repeat key terms for clarity")

    # Frequent phrases
    examples = list(sig.frequent_phrases[:5])

    # Build instruction
    instruction = _build_instruction(sig, style_notes)

    return VoicePersona(
        instruction=instruction,
        style_notes=style_notes,
        avoid=avoid,
        examples=examples,
    )


def format_ghostwrite_prompt(persona: VoicePersona, task: str) -> str:
    """Build a complete ghostwriting prompt from persona + task.

    Returns a string suitable for LLM system prompt injection.
    """
    lines = [persona.instruction, ""]

    if persona.style_notes:
        lines.append("STYLE GUIDE:")
        for note in persona.style_notes:
            lines.append(f"- {note}")

    if persona.avoid:
        lines.append("\nAVOID:")
        for a in persona.avoid:
            lines.append(f"- {a}")

    if persona.examples:
        lines.append("\nCHARACTERISTIC PHRASES:")
        for ex in persona.examples:
            lines.append(f'- "{ex}"')

    lines.append(f"\nTASK: {task}")

    return "\n".join(lines)


# --- Internal helpers ---

def _empty_signature() -> VoiceSignature:
    """Return a neutral voice signature for insufficient data."""
    return VoiceSignature(
        avg_sentence_length=15.0,
        avg_word_length=4.5,
        contraction_rate=0.15,
        question_rate=0.1,
        exclamation_rate=0.05,
        formality="neutral",
        vocabulary_richness=0.5,
        sentence_count=0,
        message_count=0,
    )


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences. Handles common abbreviations."""
    # Simple split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _classify_formality(
    contraction_rate: float,
    avg_sent_len: float,
    exclamation_rate: float,
) -> str:
    """Classify formality from linguistic markers."""
    casual_score = 0
    formal_score = 0

    if contraction_rate > 0.2:
        casual_score += 2
    elif contraction_rate < 0.05:
        formal_score += 2

    if avg_sent_len > 18:
        formal_score += 1
    elif avg_sent_len < 10:
        casual_score += 1

    if exclamation_rate > 0.15:
        casual_score += 1

    if casual_score > formal_score:
        return "casual"
    elif formal_score > casual_score:
        return "formal"
    return "neutral"


def _build_instruction(sig: VoiceSignature, style_notes: List[str]) -> str:
    """Build the main ghostwriting instruction."""
    formality_desc = {
        "casual": "conversational and relaxed",
        "formal": "professional and precise",
        "neutral": "natural and balanced",
    }
    tone = formality_desc.get(sig.formality, "natural")

    return (
        f"Write in the user's voice. Their style is {tone}, "
        f"with sentences averaging {sig.avg_sentence_length:.0f} words. "
        f"Match their tone exactly — do not add your own personality."
    )
