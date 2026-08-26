import { Grid, Eye } from 'lucide-react';

interface Props {
  imageUrl?: string;
}

export default function ConfusionMatrix({ imageUrl }: Props) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
      <div className="px-6 py-4 border-b border-slate-200 bg-slate-50/70 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Grid className="w-4 h-4 text-emerald-600" />
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
            Test Set Confusion Matrix (38 PlantVillage Classes)
          </h3>
        </div>
        <span className="text-[11px] text-slate-400 font-mono">8,145 Test Samples</span>
      </div>

      <div className="p-4 sm:p-6 flex items-center justify-center bg-slate-50/50 min-h-[360px]">
        {imageUrl ? (
          <div className="relative max-w-full overflow-auto text-center">
            <img
              src={imageUrl}
              alt="Confusion Matrix"
              className="max-h-[640px] max-w-full mx-auto rounded border border-slate-200 shadow-sm"
            />
          </div>
        ) : (
          <div className="text-slate-400 text-xs flex items-center gap-1.5">
            <Eye className="w-4 h-4" />
            <span>Confusion matrix plot not available.</span>
          </div>
        )}
      </div>
    </div>
  );
}
