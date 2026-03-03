'use client';

import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';

const DIMENSION_LABELS: Record<string, string> = {
  completeness: 'Complete',
  consistency: 'Consistent',
  coherence: 'Coherent',
  traceability: 'Traceable',
  actionability: 'Actionable',
  specificity: 'Specific',
  codegen_readiness: 'Codegen',
};

interface FidelityRadarProps {
  scores: Record<string, number>;
}

export function FidelityRadar({ scores }: FidelityRadarProps) {
  const data = Object.entries(scores).map(([key, value]) => ({
    dimension: DIMENSION_LABELS[key] || key,
    score: value,
  }));

  if (!data.length) return null;

  return (
    <div className="w-full h-64">
      <ResponsiveContainer>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
          <PolarGrid stroke="var(--border)" />
          <PolarAngleAxis
            dataKey="dimension"
            tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fill: 'var(--text-muted)', fontSize: 9 }}
          />
          <Radar
            dataKey="score"
            stroke="#6366F1"
            fill="#6366F1"
            fillOpacity={0.2}
            strokeWidth={2}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
