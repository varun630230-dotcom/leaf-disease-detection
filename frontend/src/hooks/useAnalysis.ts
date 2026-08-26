import { useState } from 'react';
import { analyzeImage } from '../services/api';

type Stage = 'validating' | 'detecting' | 'classifying' | 'locating' | 'generating' | 'done';

export const useAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [stage, setStage] = useState<Stage>('validating');
  const [error, setError] = useState<string | null>(null);
  
  const analyze = async (file: File): Promise<string | null> => {
    setIsAnalyzing(true);
    setError(null);
    setStage('validating');
    
    try {
      setTimeout(() => setStage('detecting'), 1000);
      setTimeout(() => setStage('classifying'), 2000);
      setTimeout(() => setStage('locating'), 3000);
      setTimeout(() => setStage('generating'), 4000);
      
      const id = await analyzeImage(file);
      
      setStage('done');
      setIsAnalyzing(false);
      return id;
    } catch (err: any) {
      setError(err.message || 'An error occurred during analysis.');
      setIsAnalyzing(false);
      return null;
    }
  };
  
  return { analyze, isAnalyzing, stage, error };
};
