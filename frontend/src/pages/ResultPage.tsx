import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getAnalysis } from '../services/api';
import { AnalysisResult } from '../types';
import ImageAnalysisViewer from '../components/ImageAnalysisViewer';
import PredictionSummary from '../components/PredictionSummary';
import RejectedImageState from '../components/RejectedImageState';
import UncertainState from '../components/UncertainState';
import { ArrowLeft, Loader2 } from 'lucide-react';

export default function ResultPage() {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    setError(null);

    getAnalysis(id)
      .then(data => {
        setResult(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Failed to retrieve analysis.');
        setLoading(false);
      });
  }, [id]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] text-center">
        <Loader2 className="w-8 h-8 text-emerald-600 animate-spin mb-3" />
        <p className="text-sm font-medium text-slate-600">
          Loading Visual Analysis...
        </p>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="text-center mt-12 bg-white rounded-xl border border-slate-200 p-8 shadow-sm max-w-md mx-auto">
        <p className="text-sm text-rose-600 mb-4">{error || 'Analysis record not found.'}</p>
        <Link
          to="/"
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-700 hover:text-emerald-800"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Analysis</span>
        </Link>
      </div>
    );
  }

  // Explicit state handlers
  if (result.status === 'rejected') {
    return <RejectedImageState reason={result.reason} message={result.message} />;
  }

  if (result.status === 'uncertain') {
    return <UncertainState message={result.message} />;
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Top Action Row */}
      <div className="flex items-center justify-between">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-xs sm:text-sm font-semibold text-slate-600 hover:text-slate-900 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Analyze Another Image</span>
        </Link>

        <span className="text-xs text-slate-400 font-mono">
          ID: {result.id}
        </span>
      </div>

      {/* Main 2-Column Responsive Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        {/* Left Column: Image Analysis Viewer */}
        <div className="w-full">
          {result.images && (
            <ImageAnalysisViewer
              images={result.images}
              isDiseased={result.isDiseased ?? false}
            />
          )}
        </div>

        {/* Right Column: Prediction Summary */}
        <div className="w-full">
          <PredictionSummary result={result} />
        </div>
      </div>
    </div>
  );
}
