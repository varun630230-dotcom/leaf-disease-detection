import { DiseaseKnowledge } from '../types';
import { BookOpen, AlertTriangle, ShieldCheck, Activity, Bug } from 'lucide-react';

interface Props {
  knowledge: DiseaseKnowledge;
}

export default function DiseaseKnowledgeCard({ knowledge }: Props) {
  const typeStyles: Record<string, { bg: string; text: string; border: string; icon: any }> = {
    fungal: {
      bg: 'bg-amber-50',
      text: 'text-amber-800',
      border: 'border-amber-200',
      icon: Activity,
    },
    bacterial: {
      bg: 'bg-blue-50',
      text: 'text-blue-800',
      border: 'border-blue-200',
      icon: AlertTriangle,
    },
    viral: {
      bg: 'bg-purple-50',
      text: 'text-purple-800',
      border: 'border-purple-200',
      icon: AlertTriangle,
    },
    pest_mite: {
      bg: 'bg-rose-50',
      text: 'text-rose-800',
      border: 'border-rose-200',
      icon: Bug,
    },
  };

  const style = typeStyles[knowledge.diseaseType] || typeStyles.fungal;
  const TypeIcon = style.icon;

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-5">
      {/* Header with Verified Pathogen */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <BookOpen className="w-4 h-4 text-emerald-600" />
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
              Agronomic Pathology Profile
            </h3>
          </div>
          <p className="text-xs text-slate-500 font-medium">
            Verified scientific profile for <span className="font-semibold text-slate-700">{knowledge.plant} {knowledge.disease}</span>
          </p>
        </div>

        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold uppercase tracking-wider border self-start sm:self-auto ${style.bg} ${style.text} ${style.border}`}
        >
          <TypeIcon className="w-3.5 h-3.5" />
          <span>{knowledge.diseaseType.replace('_', ' ')}</span>
        </span>
      </div>

      {/* Causative Agent / Pathogen */}
      <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 text-xs">
        <span className="font-semibold text-slate-500 uppercase tracking-wider text-[10px] block mb-0.5">
          Causative Biological Agent / Pathogen
        </span>
        <span className="italic font-serif font-bold text-slate-900 text-sm">
          {knowledge.pathogen}
        </span>
      </div>

      {/* Key Diagnostic Symptoms */}
      {knowledge.symptoms && knowledge.symptoms.length > 0 && (
        <div>
          <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
            Diagnostic Visual Indicators
          </h4>
          <ul className="space-y-1.5 text-xs text-slate-600">
            {knowledge.symptoms.map((symptom, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-emerald-600 font-bold shrink-0">•</span>
                <span className="leading-relaxed">{symptom}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Recommended Immediate Actions */}
      {knowledge.recommendedActions && knowledge.recommendedActions.length > 0 && (
        <div className="p-4 rounded-lg bg-emerald-50/50 border border-emerald-100 text-xs space-y-2">
          <div className="flex items-center gap-1.5 font-bold text-emerald-900 text-xs uppercase tracking-wider">
            <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>Recommended Agronomic Interventions</span>
          </div>
          <ul className="space-y-1.5 text-slate-700">
            {knowledge.recommendedActions.map((action, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-emerald-700 font-bold shrink-0">✓</span>
                <span className="leading-relaxed">{action}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Prevention & Risk Factors */}
      {knowledge.riskFactors && knowledge.riskFactors.length > 0 && (
        <div className="text-xs text-slate-500 border-t border-slate-100 pt-3">
          <span className="font-semibold text-slate-600">Environmental Risk Triggers: </span>
          <span>{knowledge.riskFactors.join('; ')}</span>
        </div>
      )}
    </div>
  );
}
