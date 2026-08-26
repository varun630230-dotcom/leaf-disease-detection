import { ConfidenceLevel } from '../types';

interface Props {
  level: ConfidenceLevel;
  score?: number;
}

export default function ConfidenceBadge({ level, score }: Props) {
  const styles: Record<ConfidenceLevel, { bg: string; text: string; dot: string }> = {
    HIGH: {
      bg: 'bg-emerald-50 text-emerald-800 border-emerald-200',
      text: 'HIGH',
      dot: 'bg-emerald-500',
    },
    MEDIUM: {
      bg: 'bg-amber-50 text-amber-800 border-amber-200',
      text: 'MEDIUM',
      dot: 'bg-amber-500',
    },
    LOW: {
      bg: 'bg-rose-50 text-rose-800 border-rose-200',
      text: 'LOW',
      dot: 'bg-rose-500',
    },
  };

  const current = styles[level] || styles.HIGH;

  return (
    <div className="flex items-center gap-2">
      <span
        className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-xs font-semibold uppercase tracking-wider border ${current.bg}`}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${current.dot}`} />
        {current.text}
      </span>
      {score !== undefined && (
        <span className="text-xs font-mono font-medium text-slate-600">
          {(score * 100).toFixed(1)}%
        </span>
      )}
    </div>
  );
}
