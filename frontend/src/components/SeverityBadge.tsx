import { SeverityLevel } from '../types';

interface Props {
  level: SeverityLevel;
}

export default function SeverityBadge({ level }: Props) {
  const styles: Record<SeverityLevel, { bg: string; text: string; border: string }> = {
    MINIMAL: {
      bg: 'bg-emerald-50',
      text: 'text-emerald-800',
      border: 'border-emerald-200',
    },
    MILD: {
      bg: 'bg-amber-50',
      text: 'text-amber-800',
      border: 'border-amber-200',
    },
    MODERATE: {
      bg: 'bg-orange-50',
      text: 'text-orange-800',
      border: 'border-orange-200',
    },
    SEVERE: {
      bg: 'bg-rose-50',
      text: 'text-rose-800',
      border: 'border-rose-200',
    },
  };

  const style = styles[level] || styles.MODERATE;

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-semibold uppercase tracking-wider border ${style.bg} ${style.text} ${style.border}`}
    >
      {level}
    </span>
  );
}
