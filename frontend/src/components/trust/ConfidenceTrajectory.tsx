'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ReferenceLine } from 'recharts';

interface ConfidenceTrajectoryProps {
  trajectory: number[];
}

export function ConfidenceTrajectory({ trajectory }: ConfidenceTrajectoryProps) {
  if (!trajectory.length) return null;

  const data = trajectory.map((value, i) => ({ turn: i + 1, confidence: value }));

  return (
    <div className="w-full h-48">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis
            dataKey="turn"
            tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
            label={{ value: 'Turn', position: 'bottom', fill: 'var(--text-muted)', fontSize: 10, offset: -5 }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
            width={30}
          />
          <ReferenceLine y={0.7} stroke="#10B981" strokeDasharray="3 3" strokeOpacity={0.5} />
          <ReferenceLine y={0.4} stroke="#F59E0B" strokeDasharray="3 3" strokeOpacity={0.5} />
          <Line
            type="monotone"
            dataKey="confidence"
            stroke="#6366F1"
            strokeWidth={2}
            dot={{ fill: '#6366F1', r: 3 }}
            activeDot={{ r: 5, fill: '#6366F1' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
