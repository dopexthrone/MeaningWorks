"""Messaging bridge — protocol translation between chat platforms and Motherlabs runtime.

Standalone module. No imports from core/ or engine internals.
Bridges translate platform messages (Telegram, Discord) to/from the TCP JSON
line protocol that generated agent systems speak.
"""
