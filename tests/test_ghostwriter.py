"""Tests for mother/ghostwriter.py.

Covers VoiceSignature, VoicePersona, extract_voice_signature(),
generate_voice_persona(), format_ghostwrite_prompt().
"""

import pytest
from mother.ghostwriter import (
    VoiceSignature,
    VoicePersona,
    extract_voice_signature,
    generate_voice_persona,
    format_ghostwrite_prompt,
    _empty_signature,
    _split_sentences,
    _classify_formality,
    _build_instruction,
)


# === VoiceSignature dataclass ===

class TestVoiceSignature:
    def test_frozen(self):
        sig = _empty_signature()
        with pytest.raises(AttributeError):
            sig.formality = "changed"

    def test_empty_signature_defaults(self):
        sig = _empty_signature()
        assert sig.avg_sentence_length == 15.0
        assert sig.formality == "neutral"
        assert sig.sentence_count == 0
        assert sig.message_count == 0
        assert sig.frequent_phrases == []


# === extract_voice_signature ===

class TestExtractVoiceSignature:
    def test_empty_messages(self):
        sig = extract_voice_signature([])
        assert sig.sentence_count == 0
        assert sig.message_count == 0

    def test_single_message(self):
        sig = extract_voice_signature(["Hello world."])
        assert sig.message_count == 1
        assert sig.sentence_count >= 1

    def test_multiple_messages(self):
        messages = [
            "I think we should refactor the auth module.",
            "The current implementation is too complex.",
            "Let's simplify it step by step.",
        ]
        sig = extract_voice_signature(messages)
        assert sig.message_count == 3
        assert sig.sentence_count == 3

    def test_contraction_detection(self):
        messages = [
            "I don't think that's right.",
            "We can't do it that way.",
            "It won't work.",
        ]
        sig = extract_voice_signature(messages)
        assert sig.contraction_rate > 0.0

    def test_no_contractions(self):
        messages = [
            "I do not think that is correct.",
            "We cannot proceed with that approach.",
            "It will not function properly.",
        ]
        sig = extract_voice_signature(messages)
        # May still detect some, but rate should be lower
        assert sig.contraction_rate < 0.5

    def test_question_rate(self):
        messages = [
            "What do you think?",
            "Can we do it?",
            "Is this the right approach?",
        ]
        sig = extract_voice_signature(messages)
        assert sig.question_rate > 0.5

    def test_no_questions(self):
        messages = [
            "This is the plan.",
            "We will execute it.",
            "The system works correctly.",
        ]
        sig = extract_voice_signature(messages)
        assert sig.question_rate == 0.0

    def test_exclamation_rate(self):
        messages = [
            "This is amazing!",
            "Great work!",
            "Let's ship it!",
        ]
        sig = extract_voice_signature(messages)
        assert sig.exclamation_rate > 0.5

    def test_vocabulary_richness(self):
        # All unique words → high richness
        messages = ["Alpha beta gamma delta epsilon."]
        sig = extract_voice_signature(messages)
        assert sig.vocabulary_richness > 0.5

    def test_low_vocabulary_richness(self):
        # Repeated words → low richness
        messages = ["the the the the the the the the."]
        sig = extract_voice_signature(messages)
        assert sig.vocabulary_richness < 0.5

    def test_avg_sentence_length(self):
        messages = ["One two three four five."]  # 5 words
        sig = extract_voice_signature(messages)
        assert sig.avg_sentence_length == 5.0

    def test_avg_word_length(self):
        messages = ["Hi."]  # "Hi." → word "Hi." length 3
        sig = extract_voice_signature(messages)
        assert sig.avg_word_length > 0

    def test_frequent_phrases(self):
        messages = [
            "I think we should do it. I think we must. I think we can.",
        ]
        sig = extract_voice_signature(messages)
        # "I think" should appear as frequent phrase
        assert any("think" in p for p in sig.frequent_phrases)

    def test_formality_casual(self):
        messages = [
            "I don't know! Can't figure it out!",
            "It's weird. Won't work!",
            "Let's try again! That's odd!",
        ]
        sig = extract_voice_signature(messages)
        assert sig.formality == "casual"

    def test_formality_formal(self):
        messages = [
            "The implementation demonstrates significant architectural complexity that requires careful consideration.",
            "Furthermore, the integration testing methodology should encompass all boundary conditions.",
            "Therefore, the recommendation is to proceed with comprehensive validation procedures.",
        ]
        sig = extract_voice_signature(messages)
        assert sig.formality == "formal"

    def test_whitespace_only_messages(self):
        sig = extract_voice_signature(["   ", "\n", "\t"])
        assert sig.sentence_count == 0


# === generate_voice_persona ===

class TestGenerateVoicePersona:
    def test_returns_voice_persona(self):
        sig = _empty_signature()
        persona = generate_voice_persona(sig)
        assert isinstance(persona, VoicePersona)
        assert persona.instruction
        assert isinstance(persona.style_notes, list)
        assert isinstance(persona.avoid, list)

    def test_short_sentences_noted(self):
        sig = VoiceSignature(
            avg_sentence_length=6.0, avg_word_length=4.0,
            contraction_rate=0.1, question_rate=0.1,
            exclamation_rate=0.0, formality="neutral",
            vocabulary_richness=0.5, sentence_count=10,
            message_count=5,
        )
        persona = generate_voice_persona(sig)
        notes_text = " ".join(persona.style_notes).lower()
        assert "short" in notes_text or "punchy" in notes_text

    def test_long_sentences_noted(self):
        sig = VoiceSignature(
            avg_sentence_length=25.0, avg_word_length=5.0,
            contraction_rate=0.01, question_rate=0.05,
            exclamation_rate=0.0, formality="formal",
            vocabulary_richness=0.7, sentence_count=20,
            message_count=10,
        )
        persona = generate_voice_persona(sig)
        notes_text = " ".join(persona.style_notes).lower()
        assert "longer" in notes_text or "detailed" in notes_text

    def test_high_contraction_rate(self):
        sig = VoiceSignature(
            avg_sentence_length=12.0, avg_word_length=4.0,
            contraction_rate=0.5, question_rate=0.1,
            exclamation_rate=0.1, formality="casual",
            vocabulary_richness=0.5, sentence_count=10,
            message_count=5,
        )
        persona = generate_voice_persona(sig)
        notes_text = " ".join(persona.style_notes).lower()
        assert "contraction" in notes_text

    def test_low_contraction_rate(self):
        sig = VoiceSignature(
            avg_sentence_length=15.0, avg_word_length=5.0,
            contraction_rate=0.01, question_rate=0.05,
            exclamation_rate=0.01, formality="formal",
            vocabulary_richness=0.6, sentence_count=20,
            message_count=10,
        )
        persona = generate_voice_persona(sig)
        notes_text = " ".join(persona.style_notes).lower()
        assert "avoid contraction" in notes_text or "formally" in notes_text

    def test_casual_formality(self):
        sig = VoiceSignature(
            avg_sentence_length=8.0, avg_word_length=4.0,
            contraction_rate=0.4, question_rate=0.2,
            exclamation_rate=0.3, formality="casual",
            vocabulary_richness=0.4, sentence_count=10,
            message_count=5,
        )
        persona = generate_voice_persona(sig)
        notes_text = " ".join(persona.style_notes).lower()
        assert "casual" in notes_text or "conversational" in notes_text

    def test_frequent_phrases_in_examples(self):
        sig = VoiceSignature(
            avg_sentence_length=12.0, avg_word_length=4.0,
            contraction_rate=0.1, question_rate=0.1,
            exclamation_rate=0.0, formality="neutral",
            vocabulary_richness=0.5, sentence_count=10,
            message_count=5,
            frequent_phrases=["I think", "let's go"],
        )
        persona = generate_voice_persona(sig)
        assert "I think" in persona.examples

    def test_frozen_persona(self):
        sig = _empty_signature()
        persona = generate_voice_persona(sig)
        with pytest.raises(AttributeError):
            persona.instruction = "changed"


# === format_ghostwrite_prompt ===

class TestFormatGhostwritePrompt:
    def test_contains_instruction(self):
        sig = _empty_signature()
        persona = generate_voice_persona(sig)
        prompt = format_ghostwrite_prompt(persona, "Write an email")
        assert persona.instruction in prompt

    def test_contains_task(self):
        sig = _empty_signature()
        persona = generate_voice_persona(sig)
        prompt = format_ghostwrite_prompt(persona, "Write an email to Bob")
        assert "Write an email to Bob" in prompt
        assert "TASK:" in prompt

    def test_contains_style_guide(self):
        sig = _empty_signature()
        persona = generate_voice_persona(sig)
        prompt = format_ghostwrite_prompt(persona, "task")
        assert "STYLE GUIDE:" in prompt

    def test_contains_avoid_section(self):
        sig = VoiceSignature(
            avg_sentence_length=6.0, avg_word_length=4.0,
            contraction_rate=0.5, question_rate=0.1,
            exclamation_rate=0.0, formality="casual",
            vocabulary_richness=0.5, sentence_count=10,
            message_count=5,
        )
        persona = generate_voice_persona(sig)
        prompt = format_ghostwrite_prompt(persona, "task")
        assert "AVOID:" in prompt

    def test_contains_characteristic_phrases(self):
        sig = VoiceSignature(
            avg_sentence_length=12.0, avg_word_length=4.0,
            contraction_rate=0.1, question_rate=0.1,
            exclamation_rate=0.0, formality="neutral",
            vocabulary_richness=0.5, sentence_count=10,
            message_count=5,
            frequent_phrases=["you know", "basically"],
        )
        persona = generate_voice_persona(sig)
        prompt = format_ghostwrite_prompt(persona, "task")
        assert "CHARACTERISTIC PHRASES:" in prompt
        assert "you know" in prompt


# === Internal helpers ===

class TestSplitSentences:
    def test_simple_split(self):
        sents = _split_sentences("Hello world. How are you?")
        assert len(sents) == 2

    def test_single_sentence(self):
        sents = _split_sentences("Just one sentence.")
        assert len(sents) == 1

    def test_empty_string(self):
        sents = _split_sentences("")
        assert len(sents) == 0

    def test_exclamation(self):
        sents = _split_sentences("Wow! Amazing!")
        assert len(sents) == 2


class TestClassifyFormality:
    def test_casual(self):
        assert _classify_formality(0.4, 8.0, 0.3) == "casual"

    def test_formal(self):
        assert _classify_formality(0.01, 22.0, 0.0) == "formal"

    def test_neutral(self):
        assert _classify_formality(0.1, 14.0, 0.05) == "neutral"


class TestBuildInstruction:
    def test_contains_tone(self):
        sig = _empty_signature()
        instruction = _build_instruction(sig, [])
        assert "natural" in instruction or "balanced" in instruction

    def test_contains_avg_length(self):
        sig = _empty_signature()
        instruction = _build_instruction(sig, [])
        assert "15" in instruction  # avg_sentence_length default
