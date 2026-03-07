'use client';

import { useMemo, useRef } from 'react';
import type { TaskStatusResponse } from '@/lib/api/types';

export interface ParsedProgress {
  entities: string[];
  relationships: [string, string][];
  latestInsight: string | null;
  entityTypes: Record<string, string>;
}

function guessEntityType(text: string): string {
  const lower = text.toLowerCase();
  if (lower.includes('service') || lower.includes('manager') || lower.includes('handler')) return 'process';
  if (lower.includes('api') || lower.includes('endpoint') || lower.includes('interface') || lower.includes('gateway')) return 'interface';
  if (lower.includes('event') || lower.includes('notification') || lower.includes('webhook')) return 'event';
  if (lower.includes('rule') || lower.includes('constraint') || lower.includes('policy') || lower.includes('validation')) return 'constraint';
  if (lower.includes('system') || lower.includes('module') || lower.includes('subsystem')) return 'subsystem';
  return 'entity';
}

function extractEntitiesFromText(text: string): string[] {
  // "Extracted: X, Y, Z" or "Identified: X, Y, Z"
  const extractMatch = text.match(/^(?:Extracted|Identified):\s*(.+)$/i);
  if (extractMatch) {
    return extractMatch[1].split(/,\s*/).map(s => s.trim()).filter(Boolean);
  }

  // "Canonical set applied: X, Y, Z" or "Canonical set from input: N components"
  const canonicalMatch = text.match(/^Canonical set (?:applied|from input):\s*(.+)$/i);
  if (canonicalMatch) {
    const val = canonicalMatch[1];
    // Skip "5 components" style summaries
    if (/^\d+\s+components?$/i.test(val)) return [];
    return val.split(/,\s*/).map(s => s.trim()).filter(Boolean);
  }

  return [];
}

/** Clean raw insight text for display — strip arrow prefixes, stage markers, brackets */
function cleanInsightForDisplay(text: string): string {
  let cleaned = text
    .replace(/^\s*→?\s*/, '')           // Strip arrow prefix
    .replace(/^\s*◇\s*/, '')            // Strip diamond stage marker
    .replace(/^\[[\w]+\]\s*/, '')        // Strip [stageName] prefix
    .replace(/^\s*⚡\s*/, '')            // Strip lightning
    .replace(/^\s*⚠\s*/, '')            // Strip warning
    .trim();

  // Skip noisy internal messages
  if (cleaned.startsWith('Stage:') || cleaned.startsWith('Staged pipeline:')) {
    return '';
  }

  return cleaned;
}

function shouldSkipForEntities(text: string): boolean {
  if (text.startsWith('Core need:')) return true;
  // Skip persona-style lines (e.g., "Alex Chen: perspective...")
  if (text.includes(': ') && text.split(': ')[0].split(' ').length <= 3) return true;
  return false;
}

export function useInsightParser(pollData: TaskStatusResponse | null): ParsedProgress {
  // Use ref for accumulation to keep entities append-only across renders
  const accumulatedEntities = useRef<string[]>([]);
  const accumulatedTypes = useRef<Record<string, string>>({});

  return useMemo(() => {
    if (!pollData?.progress) {
      return {
        entities: accumulatedEntities.current,
        relationships: [],
        latestInsight: null,
        entityTypes: accumulatedTypes.current,
      };
    }

    const progress = pollData.progress;
    const structured = progress.structured_insights ?? [];
    const flat = progress.insights ?? [];

    // Extract entities from discovery insights
    const seen = new Set(accumulatedEntities.current);

    for (const si of structured) {
      if (si.category === 'discovery') {
        if (shouldSkipForEntities(si.text)) continue;
        const extracted = extractEntitiesFromText(si.text);
        for (const name of extracted) {
          if (!seen.has(name)) {
            seen.add(name);
            accumulatedEntities.current = [...accumulatedEntities.current, name];
            accumulatedTypes.current = { ...accumulatedTypes.current, [name]: guessEntityType(name) };
          }
        }
      }
    }

    // Also check flat insights for entity extraction patterns
    for (const f of flat) {
      const cleaned = f.replace(/^\s*→?\s*/, '');
      if (shouldSkipForEntities(cleaned)) continue;
      const extracted = extractEntitiesFromText(cleaned);
      for (const name of extracted) {
        if (!seen.has(name)) {
          seen.add(name);
          accumulatedEntities.current = [...accumulatedEntities.current, name];
          accumulatedTypes.current = { ...accumulatedTypes.current, [name]: guessEntityType(name) };
        }
      }
    }

    // Latest insight: prefer structured, fall back to flat. Clean for display.
    let latestInsight: string | null = null;
    // Walk backwards through structured insights to find a displayable one
    for (let i = structured.length - 1; i >= 0; i--) {
      const cleaned = cleanInsightForDisplay(structured[i].text);
      if (cleaned) { latestInsight = cleaned; break; }
    }
    // If no structured insight, try flat insights
    if (!latestInsight) {
      for (let i = flat.length - 1; i >= 0; i--) {
        const cleaned = cleanInsightForDisplay(flat[i]);
        if (cleaned) { latestInsight = cleaned; break; }
      }
    }

    // Build sequential relationships from entities
    const entities = accumulatedEntities.current;
    const relationships: [string, string][] = [];
    for (let i = 0; i < entities.length - 1; i++) {
      relationships.push([entities[i], entities[i + 1]]);
    }

    return {
      entities,
      relationships,
      latestInsight,
      entityTypes: accumulatedTypes.current,
    };
  }, [pollData]);
}
