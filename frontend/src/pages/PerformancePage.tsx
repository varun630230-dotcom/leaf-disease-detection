import { useEffect, useState } from 'react';
import { getPerformance } from '../services/api';
import { PerformanceData } from '../types';
import PerformanceMetrics from '../components/PerformanceMetrics';
import ConfusionMatrix from '../components/ConfusionMatrix';
import MetricsTable from '../components/MetricsTable';
import { Activity, AlertOctagon, Cpu, Loader2 } from 'lucide-react';

export default function PerformancePage() {
  const [data, setData] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPerformance()
      .then(res => {
        setData(res);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Failed to load model performance data.');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] text-center">
        <Loader2 className="w-8 h-8 text-emerald-600 animate-spin mb-3" />
        <p className="text-sm font-medium text-slate-600">
          Loading Evaluation Benchmarks...
        </p>
      </div>
    );
  }

  if (error || !data || data.status === 'not_evaluated') {
    return (
      <div className="text-center mt-12 bg-white rounded-xl border border-slate-200 p-10 shadow-sm max-w-lg mx-auto">
        <Activity className="w-8 h-8 text-slate-400 mx-auto mb-3" />
        <h2 className="text-lg font-bold text-slate-900 mb-1">Model Not Evaluated</h2>
        <p className="text-xs text-slate-500">
          {error || 'Model benchmark report has not been generated yet.'}
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-10">
      {/* Header */}
      <div className="space-y-1">
        <div className="flex items-center gap-2 text-xs font-bold text-emerald-700 uppercase tracking-wider">
          <Cpu className="w-4 h-4" />
          <span>Model Evaluation & Benchmarks</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Performance & Technical Validation
        </h1>
        <p className="text-xs sm:text-sm text-slate-500">
          Empirical evaluation on PlantVillage held-out test split (8,145 samples) across 38 crop and disease classes.
        </p>
      </div>

      {/* 1. Core Metrics Grid (Overall, OOD, Segmentation, Latency, Model Comparison) */}
      <PerformanceMetrics
        overall={data.overall}
        ood={data.ood}
        segmentation={data.segmentation}
        latency={data.latency}
        modelComparison={data.modelComparison}
      />

      {/* 2. Confusion Matrix Plot */}
      <ConfusionMatrix imageUrl={data.confusionMatrixUrl} />

      {/* 3. Per-Class Sortable Breakdown Table */}
      {data.perClass && data.perClass.length > 0 && (
        <MetricsTable data={data.perClass} />
      )}

      {/* 4. Known Technical Limitations */}
      {data.limitations && data.limitations.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-900 uppercase tracking-wider">
            <AlertOctagon className="w-4 h-4 text-amber-600" />
            <span>Known Operational Limitations</span>
          </div>
          <ul className="space-y-2 text-xs text-slate-600">
            {data.limitations.map((lim, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-amber-500 font-bold">•</span>
                <span className="leading-relaxed">{lim}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
