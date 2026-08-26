import { useEffect, useState } from 'react';
import { Check, Loader2 } from 'lucide-react';

export default function AnalysisLoader() {
  const stages = [
    'Validating image',
    'Detecting leaf',
    'Identifying plant',
    'Classifying disease',
    'Locating affected regions',
    'Estimating severity',
    'Generating explanation',
  ];

  const [currentStageIdx, setCurrentStageIdx] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStageIdx(prev => (prev < stages.length - 1 ? prev + 1 : prev));
    }, 450);

    return () => clearInterval(interval);
  }, [stages.length]);

  return (
    <div className="w-full max-w-md mx-auto bg-white rounded-xl border border-slate-200 shadow-sm p-8 text-left">
      <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-100">
        <Loader2 className="w-5 h-5 text-emerald-600 animate-spin shrink-0" />
        <h3 className="text-sm font-semibold text-slate-900">
          Running Neural Vision Pipeline
        </h3>
      </div>

      <div className="space-y-3.5">
        {stages.map((stage, idx) => {
          const isDone = idx < currentStageIdx;
          const isCurrent = idx === currentStageIdx;

          return (
            <div
              key={idx}
              className={`flex items-center gap-3 text-xs transition-colors duration-200 ${
                isDone
                  ? 'text-slate-900 font-medium'
                  : isCurrent
                  ? 'text-emerald-700 font-semibold'
                  : 'text-slate-400'
              }`}
            >
              <div
                className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] shrink-0 transition-all ${
                  isDone
                    ? 'bg-emerald-100 text-emerald-700'
                    : isCurrent
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-100 text-slate-400'
                }`}
              >
                {isDone ? (
                  <Check className="w-3 h-3 stroke-[3]" />
                ) : isCurrent ? (
                  <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                ) : (
                  <span>{idx + 1}</span>
                )}
              </div>
              <span>{stage}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
