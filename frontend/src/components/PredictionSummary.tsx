import { AnalysisResult } from '../types';
import SeverityBadge from './SeverityBadge';
import ConfidenceBadge from './ConfidenceBadge';
import { ChevronDown, Sparkles, Cpu } from 'lucide-react';

interface Props {
  result: AnalysisResult;
}

export default function PredictionSummary({ result }: Props) {
  const isDiseased = result.isDiseased ?? false;
  const affectedPercent =
    result.affectedAreaPercentage !== undefined
      ? (result.affectedAreaPercentage * 100).toFixed(1)
      : null;

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 sm:p-7 flex flex-col gap-6">
      {/* 1 & 2: Plant & Health Status */}
      <div>
        <div className="flex items-center justify-between gap-3 mb-1.5">
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
            {result.plant || 'Unknown Plant'}
          </h2>
          {isDiseased ? (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded text-xs font-bold uppercase tracking-wider bg-rose-50 text-rose-700 border border-rose-200">
              DISEASED
            </span>
          ) : (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded text-xs font-bold uppercase tracking-wider bg-emerald-50 text-emerald-700 border border-emerald-200">
              HEALTHY
            </span>
          )}
        </div>

        {/* 3: Disease Name / Healthy Status */}
        <p className="text-lg font-medium text-slate-700">
          {isDiseased ? result.disease : 'No disease detected.'}
        </p>
      </div>

      {/* 4 & 5: Severity & Affected Area (Diseased Only) */}
      {isDiseased && (
        <div className="grid grid-cols-2 gap-4 py-4 border-y border-slate-100 bg-slate-50/50 -mx-6 sm:-mx-7 px-6 sm:px-7">
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
              Severity
            </p>
            <SeverityBadge level={result.severity || 'MODERATE'} />
            {result.severityDescription && (
              <p className="text-[11px] text-slate-500 mt-1 line-clamp-2">
                {result.severityDescription}
              </p>
            )}
          </div>

          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
              Affected Area
            </p>
            <div className="flex items-center gap-2.5">
              <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-rose-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(100, Math.max(0, parseFloat(affectedPercent || '0')))}%` }}
                />
              </div>
              <span className="text-sm font-bold text-slate-900 shrink-0 font-mono">
                {affectedPercent}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Confidence */}
      <div>
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
          Confidence
        </p>
        <ConfidenceBadge level={result.confidenceLevel} score={result.confidence} />
      </div>

      {/* 6, 7 & 8: Concise Visual Analysis */}
      {result.visualAnalysis && (
        <div className="p-4 rounded-lg bg-emerald-50/40 border border-emerald-100 text-xs text-slate-700 space-y-1.5">
          <div className="flex items-center gap-1.5 text-emerald-900 font-semibold text-xs">
            <Sparkles className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
            <span>VISUAL ANALYSIS</span>
          </div>
          <p className="leading-relaxed">
            {result.visualAnalysis}
          </p>
        </div>
      )}

      {/* Top Alternative Predictions Dropdown */}
      {result.topPredictions && result.topPredictions.length > 1 && (
        <div>
          <details className="text-xs group">
            <summary className="cursor-pointer text-slate-500 hover:text-slate-900 font-medium list-none flex items-center justify-between py-1.5 border-t border-slate-100">
              <span>Alternative Predictions</span>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 group-open:rotate-180 transition-transform" />
            </summary>
            <ul className="mt-2 space-y-1.5 text-slate-600 pl-1">
              {result.topPredictions.slice(1, 4).map((p, i) => (
                <li key={i} className="flex justify-between items-center py-0.5">
                  <span className="truncate pr-2">
                    {p.plant} {p.disease ? `— ${p.disease}` : ''}
                  </span>
                  <span className="font-mono text-slate-700 shrink-0">
                    {(p.confidence * 100).toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          </details>
        </div>
      )}

      {/* Footer Model Timing */}
      {result.modelInfo && (
        <div className="mt-auto pt-3 border-t border-slate-100 text-[11px] text-slate-400 flex justify-between items-center">
          <span className="flex items-center gap-1">
            <Cpu className="w-3 h-3 text-slate-300" />
            Model {result.modelInfo.version}
          </span>
          <span>{result.modelInfo.inferenceTime} ms</span>
        </div>
      )}
    </div>
  );
}
