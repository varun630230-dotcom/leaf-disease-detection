import {
  OverallMetrics,
  OODMetrics,
  SegmentationMetrics,
  LatencyMetrics,
  ModelComparisonItem,
} from '../types';
import { Activity, ShieldCheck, Layers, Zap, CheckCircle2 } from 'lucide-react';

interface Props {
  overall?: OverallMetrics;
  ood?: OODMetrics;
  segmentation?: SegmentationMetrics;
  latency?: LatencyMetrics;
  modelComparison?: ModelComparisonItem[];
}

export default function PerformanceMetrics({
  overall,
  ood,
  segmentation,
  latency,
  modelComparison,
}: Props) {
  return (
    <div className="space-y-8">
      {/* 1. Overall Classification Metrics (4-column grid) */}
      <div>
        <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4 text-emerald-600" />
          <span>Classification Performance (38 Classes)</span>
        </h3>

        {overall ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
                Accuracy
              </p>
              <p className="text-2xl sm:text-3xl font-extrabold text-slate-900 font-mono">
                {(overall.accuracy * 100).toFixed(1)}%
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Test Set (8,145 samples)</p>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
                Precision
              </p>
              <p className="text-2xl sm:text-3xl font-extrabold text-slate-900 font-mono">
                {(overall.precision * 100).toFixed(1)}%
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Macro Average</p>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
                Recall
              </p>
              <p className="text-2xl sm:text-3xl font-extrabold text-slate-900 font-mono">
                {(overall.recall * 100).toFixed(1)}%
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Macro Average</p>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
                F1 Score
              </p>
              <p className="text-2xl sm:text-3xl font-extrabold text-emerald-600 font-mono">
                {(overall.f1 * 100).toFixed(1)}%
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Balanced Metric</p>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-xl border border-slate-200 p-6 text-center text-sm text-slate-500">
            Not evaluated
          </div>
        )}
      </div>

      {/* 2. Secondary Metrics (OOD, Segmentation, Latency) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* OOD Evaluation */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-slate-900 font-bold text-xs uppercase tracking-wider border-b border-slate-100 pb-2">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span>OOD Detection (Free Energy)</span>
          </div>

          {ood ? (
            <div className="space-y-2.5 text-xs">
              <div className="flex justify-between items-center py-1">
                <span className="text-slate-500">OOD AUROC</span>
                <span className="font-mono font-bold text-slate-900">{(ood.auroc * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center py-1 border-t border-slate-50">
                <span className="text-slate-500">False Positive Rate (FPR@95)</span>
                <span className="font-mono font-bold text-slate-900">{(ood.fpr95 * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center py-1 border-t border-slate-50">
                <span className="text-slate-500">Non-Leaf Rejection Rate</span>
                <span className="font-mono font-bold text-emerald-700">{(ood.rejectionRate * 100).toFixed(1)}%</span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">Not evaluated</p>
          )}
        </div>

        {/* Segmentation */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-slate-900 font-bold text-xs uppercase tracking-wider border-b border-slate-100 pb-2">
            <Layers className="w-4 h-4 text-blue-600" />
            <span>Lesion Segmentation</span>
          </div>

          {segmentation ? (
            <div className="space-y-2.5 text-xs">
              <div className="flex justify-between items-center py-1">
                <span className="text-slate-500">Mean IoU (Intersection over Union)</span>
                <span className="font-mono font-bold text-slate-900">{segmentation.meanIoU.toFixed(3)}</span>
              </div>
              <div className="flex justify-between items-center py-1 border-t border-slate-50">
                <span className="text-slate-500">Dice Similarity Coefficient</span>
                <span className="font-mono font-bold text-slate-900">{segmentation.diceScore.toFixed(3)}</span>
              </div>
              <p className="text-[11px] text-slate-400 pt-1">Weakly-supervised Otsu lesion localization</p>
            </div>
          ) : (
            <p className="text-xs text-slate-400">Not evaluated</p>
          )}
        </div>

        {/* Inference Latency */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-slate-900 font-bold text-xs uppercase tracking-wider border-b border-slate-100 pb-2">
            <Zap className="w-4 h-4 text-amber-600" />
            <span>Inference Latency (CPU)</span>
          </div>

          {latency ? (
            <div className="space-y-2.5 text-xs">
              <div className="flex justify-between items-center py-1">
                <span className="text-slate-500">Average Latency</span>
                <span className="font-mono font-bold text-slate-900">{latency.meanMs.toFixed(1)} ms</span>
              </div>
              <div className="flex justify-between items-center py-1 border-t border-slate-50">
                <span className="text-slate-500">P50 / P95 Percentile</span>
                <span className="font-mono font-bold text-slate-900">{latency.p50Ms.toFixed(1)} / {latency.p95Ms.toFixed(1)} ms</span>
              </div>
              <div className="flex justify-between items-center py-1 border-t border-slate-50">
                <span className="text-slate-500">Model Size</span>
                <span className="font-mono font-bold text-slate-900">{latency.modelSizeMb.toFixed(1)} MB</span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">Not evaluated</p>
          )}
        </div>
      </div>

      {/* 3. Model Comparison Table */}
      {modelComparison && modelComparison.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
          <div className="px-6 py-4 border-b border-slate-200 bg-slate-50/70">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              Architecture Benchmark & Model Selection
            </h4>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-500 uppercase font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-3 px-6">Model Architecture</th>
                  <th className="py-3 px-6">Accuracy</th>
                  <th className="py-3 px-6">Macro F1</th>
                  <th className="py-3 px-6">Latency (CPU)</th>
                  <th className="py-3 px-6">Size</th>
                  <th className="py-3 px-6">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {modelComparison.map((item, idx) => (
                  <tr
                    key={idx}
                    className={item.isSelected ? 'bg-emerald-50/40 font-medium' : 'hover:bg-slate-50/60'}
                  >
                    <td className="py-3 px-6 text-slate-900 font-semibold flex items-center gap-1.5">
                      {item.isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />}
                      <span>{item.model}</span>
                    </td>
                    <td className="py-3 px-6 font-mono">{(item.accuracy * 100).toFixed(1)}%</td>
                    <td className="py-3 px-6 font-mono">{(item.macroF1 * 100).toFixed(1)}%</td>
                    <td className="py-3 px-6 font-mono">{item.meanLatencyMs.toFixed(1)} ms</td>
                    <td className="py-3 px-6 font-mono">{item.modelSizeMb.toFixed(1)} MB</td>
                    <td className="py-3 px-6">
                      {item.isSelected ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-800">
                          DEPLOYED
                        </span>
                      ) : (
                        <span className="text-slate-400 text-[10px]">Evaluated</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
