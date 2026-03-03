"""Tests for StreamingVoiceTracker — progressive voice during token streaming."""

import pytest

from mother.screens.chat import StreamingVoiceTracker, _extract_first_sentence


# --- _extract_first_sentence helper ---


class TestExtractFirstSentence:
    def test_complete_sentence(self):
        assert _extract_first_sentence("This is a long sentence. More text") == "This is a long sentence."

    def test_exclamation(self):
        assert _extract_first_sentence("This is exciting! Really") == "This is exciting!"

    def test_question(self):
        assert _extract_first_sentence("Is this working? Yes") == "Is this working?"

    def test_too_short(self):
        """Periods before position 10 are ignored (abbreviations etc.)."""
        assert _extract_first_sentence("Hi. More.") is None

    def test_no_boundary(self):
        assert _extract_first_sentence("Still going with no end") is None

    def test_empty(self):
        assert _extract_first_sentence("") is None


# --- StreamingVoiceTracker ---


class TestStreamingVoiceTracker:
    def test_empty_feed(self):
        tracker = StreamingVoiceTracker()
        assert tracker.feed("") is None

    def test_detects_voice_tag(self):
        tracker = StreamingVoiceTracker()
        tracker.feed("[VOICE]Hello world.")
        assert tracker.voice_detected is True

    def test_no_voice_tags_detected_false(self):
        tracker = StreamingVoiceTracker()
        tracker.feed("Just plain text without any tags.")
        assert tracker.voice_detected is False

    def test_no_voice_tags_spoke_nothing(self):
        tracker = StreamingVoiceTracker()
        tracker.feed("Just plain text without any tags.")
        assert tracker.spoke_anything is False

    def test_spoke_anything_false_initially(self):
        tracker = StreamingVoiceTracker()
        assert tracker.spoke_anything is False

    def test_extracts_first_sentence(self):
        tracker = StreamingVoiceTracker()
        result = tracker.feed("[VOICE]This is a complete sentence. More coming")
        assert result == "This is a complete sentence."

    def test_spoke_anything_true_after_sentence(self):
        tracker = StreamingVoiceTracker()
        tracker.feed("[VOICE]This is a complete sentence. More coming")
        assert tracker.spoke_anything is True

    def test_returns_none_incomplete(self):
        tracker = StreamingVoiceTracker()
        result = tracker.feed("[VOICE]Still going with no end yet")
        assert result is None

    def test_handles_close_tag(self):
        tracker = StreamingVoiceTracker()
        # First feed opens the voice block
        tracker.feed("[VOICE]Hello world")
        # Second feed closes it
        result = tracker.feed("[/VOICE]")
        assert result == "Hello world"

    def test_finish_returns_unspoken(self):
        tracker = StreamingVoiceTracker()
        tracker.feed("[VOICE]Partial sentence no ending")
        result = tracker.finish()
        assert result == "Partial sentence no ending"

    def test_finish_none_when_empty(self):
        tracker = StreamingVoiceTracker()
        tracker.feed("[VOICE]This is a complete sentence.")
        # The close tag finalizes and returns the sentence
        tracker.feed("[/VOICE]")
        result = tracker.finish()
        assert result is None

    def test_finish_none_no_voice(self):
        tracker = StreamingVoiceTracker()
        tracker.feed("no voice tags here")
        assert tracker.finish() is None

    def test_multiple_sentences_sequential(self):
        tracker = StreamingVoiceTracker()
        # First sentence
        r1 = tracker.feed("[VOICE]First sentence here. Second sentence here. Tail")
        assert r1 == "First sentence here."
        # Feed more to trigger second sentence extraction
        r2 = tracker.feed(" text to continue.")
        # The second sentence should have been extractable
        # "Second sentence here." is in the buffer already
        # After first extraction, unspoken starts at "Second sentence here. Tail text to continue."
        # _extract_first_sentence should find "Second sentence here."
        assert r2 == "Second sentence here."

    def test_split_tag_across_tokens(self):
        tracker = StreamingVoiceTracker()
        r1 = tracker.feed("[VOI")
        assert r1 is None
        assert tracker.voice_detected is False
        r2 = tracker.feed("CE]This is complete. More")
        assert tracker.voice_detected is True
        assert r2 == "This is complete."

    def test_action_tags_ignored(self):
        tracker = StreamingVoiceTracker()
        result = tracker.feed("[ACTION:compile]build a chat app[/ACTION]")
        assert result is None
        assert tracker.voice_detected is False

    def test_token_by_token_feeding(self):
        """Feed character by character through a full response."""
        tracker = StreamingVoiceTracker()
        text = "[VOICE]This is a sentence. Another one.[/VOICE]"
        results = []
        for ch in text:
            r = tracker.feed(ch)
            if r:
                results.append(r)
        # Should have extracted at least one sentence during streaming
        assert len(results) >= 1
        assert results[0] == "This is a sentence."

    def test_voice_close_returns_tail(self):
        tracker = StreamingVoiceTracker()
        tracker.feed("[VOICE]hello world")
        result = tracker.feed("[/VOICE]")
        assert result == "hello world"

    def test_multiple_voice_blocks(self):
        """First [VOICE] block works; tracker enters/exits correctly."""
        tracker = StreamingVoiceTracker()
        r1 = tracker.feed("[VOICE]First block.[/VOICE]")
        assert r1 == "First block."
        # After close, tracker is no longer in_voice
        # A second [VOICE] block should also work
        r2 = tracker.feed(" some text [VOICE]Second block.[/VOICE]")
        assert r2 == "Second block."

    def test_sentence_boundary_at_end_of_text(self):
        """Sentence ending right at end of accumulated text (no trailing space)."""
        tracker = StreamingVoiceTracker()
        r = tracker.feed("[VOICE]This ends cleanly.")
        # _extract_first_sentence requires trailing space or end
        # "rest = text[i+1:]" → rest is "", so not rest is True → should match
        assert r == "This ends cleanly."

    def test_finish_after_all_spoken(self):
        """finish() returns None when everything was already spoken."""
        tracker = StreamingVoiceTracker()
        tracker.feed("[VOICE]Complete sentence.")
        # Sentence was spoken during feed
        tail = tracker.finish()
        # Nothing left — already spoke "Complete sentence." and voice block unclosed
        # Actually: the unclosed block means finish() checks unspoken portion
        # spoken_len advanced past "Complete sentence." so remaining is empty
        assert tail is None


class TestStreamingVoiceTrackerEdgeCases:
    def test_empty_voice_block(self):
        tracker = StreamingVoiceTracker()
        result = tracker.feed("[VOICE][/VOICE]")
        assert result is None
        assert tracker.voice_detected is True

    def test_whitespace_only_voice_block(self):
        tracker = StreamingVoiceTracker()
        result = tracker.feed("[VOICE]   [/VOICE]")
        assert result is None

    def test_voice_tag_in_middle_of_text(self):
        tracker = StreamingVoiceTracker()
        r1 = tracker.feed("Some preamble [VOICE]Spoken part.")
        # The tag is found in accumulated raw, spoken part extracted
        assert r1 == "Spoken part."
        assert tracker.voice_detected is True

    def test_feed_after_close(self):
        """Feeding tokens after a voice block closes should look for next [VOICE]."""
        tracker = StreamingVoiceTracker()
        tracker.feed("[VOICE]First.[/VOICE]")
        # Now not in voice mode, feed more
        r = tracker.feed(" gap [VOICE]Second.[/VOICE]")
        assert r == "Second."
