import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { analyzeImage, getSupportedPlants } from '../services/api';
import ImageUploader from '../components/ImageUploader';
import AnalysisLoader from '../components/AnalysisLoader';
import { Leaf, ShieldAlert } from 'lucide-react';

export default function AnalyzePage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [supportedPlants, setSupportedPlants] = useState<string[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    getSupportedPlants()
      .then(plants => setSupportedPlants(plants))
      .catch(() => {});
  }, []);

  const handleImageSelected = async (file: File) => {
    setIsAnalyzing(true);
    setError(null);

    try {
      const analysisId = await analyzeImage(file);
      navigate(`/result/${analysisId}`);
    } catch (err: any) {
      setIsAnalyzing(false);
      setError(err.message || 'An unexpected error occurred during analysis.');
    }
  };

  return (
    <div className="max-w-4xl mx-auto text-center py-6 sm:py-10 space-y-10">
      {/* Hero Header */}
      <div className="space-y-2">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          LeafGuard AI
        </h1>
        <p className="text-sm sm:text-base text-slate-500 font-medium max-w-lg mx-auto">
          Plant Disease Detection & Visual Analysis
        </p>
      </div>

      {/* Main Upload / Loader Section */}
      <div className="flex justify-center">
        {isAnalyzing ? (
          <AnalysisLoader />
        ) : (
          <ImageUploader
            onImageSelected={handleImageSelected}
            isAnalyzing={isAnalyzing}
          />
        )}
      </div>

      {/* Error alert */}
      {error && !isAnalyzing && (
        <div className="max-w-md mx-auto p-4 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-start gap-2.5 text-left">
          <ShieldAlert className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <p className="leading-relaxed">{error}</p>
        </div>
      )}

      {/* Supported Agricultural Crops */}
      {supportedPlants.length > 0 && !isAnalyzing && (
        <div className="pt-6 border-t border-slate-200/60 max-w-2xl mx-auto">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2.5 flex items-center justify-center gap-1.5">
            <Leaf className="w-3.5 h-3.5 text-emerald-600" />
            <span>Supported Crops</span>
          </p>
          <p className="text-xs text-slate-500 leading-relaxed">
            {supportedPlants.join(' • ')}
          </p>
        </div>
      )}
    </div>
  );
}
