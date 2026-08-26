import os

base_dir = r"C:\Users\varun\.gemini\antigravity\scratch\leafguard-ai\frontend"

files = {
    "package.json": """{
  "name": "leafguard-frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.21.0",
    "recharts": "^2.10.3",
    "lucide-react": "^0.300.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.2.2",
    "vite": "^5.0.8"
  }
}""",
    "tsconfig.json": """{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}""",
    "tsconfig.node.json": """{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}""",
    "vite.config.ts": """import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
});""",
    "tailwind.config.js": """/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        primary: {
          DEFAULT: '#111827',
        },
        secondary: {
          DEFAULT: '#6b7280',
        },
        accent: {
          DEFAULT: '#16a34a',
        }
      }
    },
  },
  plugins: [],
}""",
    "postcss.config.js": """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}""",
    "index.html": """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <title>LeafGuard AI</title>
  </head>
  <body class="bg-gray-50 text-gray-900 font-sans antialiased">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>""",
    "src/main.tsx": """import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);""",
    "src/index.css": """@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply bg-gray-50 text-gray-900;
  }
}""",
    "src/App.tsx": """import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import AnalyzePage from './pages/AnalyzePage';
import ResultPage from './pages/ResultPage';
import PerformancePage from './pages/PerformancePage';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<AnalyzePage />} />
          <Route path="/result/:id" element={<ResultPage />} />
          <Route path="/performance" element={<PerformancePage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;""",
    "src/types/index.ts": """export interface AnalysisResult {
  id: string;
  status: 'processing' | 'success' | 'rejected' | 'uncertain' | 'error';
  plant?: string;
  disease?: string;
  isDiseased?: boolean;
  severity?: 'MINIMAL' | 'MILD' | 'MODERATE' | 'SEVERE';
  affectedAreaPercentage?: number;
  confidence: number;
  confidenceLevel: 'HIGH' | 'MEDIUM' | 'LOW';
  topPredictions?: Array<{
    plant: string;
    disease: string;
    confidence: number;
  }>;
  images?: {
    original: string;
    mask?: string;
    gradcam?: string;
    overlay?: string;
  };
  modelInfo?: {
    version: string;
    inferenceTime: number;
  };
  errorMessage?: string;
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface ClassPerformance {
  plant: string;
  disease: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface PerformanceData {
  modelInfo: {
    architecture: string;
    version: string;
    dataset: string;
    classes: number;
    size: string;
  };
  overallMetrics: Metrics;
  perClassPerformance: ClassPerformance[];
  oodMetrics: {
    auroc: number;
    fpr95: number;
    rejectionRate: number;
  };
  inferencePerformance: {
    meanLatency: number;
    p50Latency: number;
    p95Latency: number;
  };
  confusionMatrixUrl: string;
  limitations: string[];
}""",
    "src/services/api.ts": """import { AnalysisResult, PerformanceData } from '../types';

const API_BASE = '/api';

export const analyzeImage = async (file: File): Promise<string> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Analysis failed');
  const data = await res.json();
  return data.id;
};

export const getAnalysis = async (id: string): Promise<AnalysisResult> => {
  const res = await fetch(`${API_BASE}/result/${id}`);
  if (!res.ok) throw new Error('Failed to fetch result');
  return res.json();
};

export const getPerformance = async (): Promise<PerformanceData> => {
  const res = await fetch(`${API_BASE}/performance`);
  if (!res.ok) throw new Error('Failed to fetch performance data');
  return res.json();
};

export const getSupportedPlants = async (): Promise<string[]> => {
  const res = await fetch(`${API_BASE}/plants`);
  if (!res.ok) return ['Apple', 'Cherry', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato'];
  return res.json();
};""",
    "src/hooks/useAnalysis.ts": """import { useState } from 'react';
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
};""",
    "src/utils/format.ts": """export const formatPercent = (val: number) => `${(val * 100).toFixed(1)}%`;
export const formatMs = (val: number) => `${val.toFixed(0)}ms`;
export const formatNumber = (val: number) => new Intl.NumberFormat('en-US').format(val);""",
    "src/components/Layout.tsx": """import { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Leaf, BarChart2 } from 'lucide-react';

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 text-primary font-semibold text-lg">
            <Leaf className="w-6 h-6 text-accent" />
            LeafGuard AI
          </Link>
          <nav>
            <Link to="/performance" className="text-secondary hover:text-primary flex items-center gap-1 text-sm font-medium transition-colors">
              <BarChart2 className="w-4 h-4" />
              Performance
            </Link>
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {children}
      </main>
    </div>
  );
}""",
    "src/components/ImageUploader.tsx": """import React, { useRef, useState } from 'react';
import { Upload, X } from 'lucide-react';

interface Props {
  onImageSelected: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
}

export default function ImageUploader({ onImageSelected, selectedFile, onClear }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSelect(e.dataTransfer.files[0]);
    }
  };
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSelect(e.target.files[0]);
    }
  };
  
  const validateAndSelect = (file: File) => {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      alert('Invalid file type. Only JPG, PNG, and WEBP are supported.');
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      alert('File is too large. Maximum size is 25MB.');
      return;
    }
    onImageSelected(file);
  };
  
  if (selectedFile) {
    const previewUrl = URL.createObjectURL(selectedFile);
    return (
      <div className="border border-gray-200 rounded-lg p-4 bg-white flex items-center gap-4 shadow-sm">
        <div className="w-16 h-16 rounded-md overflow-hidden bg-gray-100 flex-shrink-0">
          <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 truncate">{selectedFile.name}</p>
          <p className="text-xs text-gray-500">{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</p>
        </div>
        <button onClick={onClear} className="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100 transition-colors">
          <X className="w-5 h-5" />
        </button>
      </div>
    );
  }
  
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        isDragging ? 'border-accent bg-green-50' : 'border-gray-300 hover:border-gray-400 bg-white'
      }`}
    >
      <input type="file" ref={fileInputRef} onChange={handleChange} accept=".jpg,.jpeg,.png,.webp" className="hidden" />
      <Upload className="w-10 h-10 mx-auto text-gray-400 mb-4" />
      <p className="text-base font-medium text-gray-900 mb-1">Drag & drop a leaf image here</p>
      <p className="text-sm text-gray-500 mb-4">or</p>
      <div className="flex justify-center gap-3">
        <button onClick={() => fileInputRef.current?.click()} className="px-4 py-2 bg-white border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors">
          Browse Files
        </button>
      </div>
      <p className="mt-4 text-xs text-gray-500">JPG, PNG, WEBP &middot; Max 25 MB</p>
    </div>
  );
}""",
    "src/components/AnalysisLoading.tsx": """import { CheckCircle2, Circle, Loader2 } from 'lucide-react';

const STAGES = [
  { id: 'validating', label: 'Validating image...' },
  { id: 'detecting', label: 'Detecting leaf...' },
  { id: 'classifying', label: 'Classifying disease...' },
  { id: 'locating', label: 'Locating affected regions...' },
  { id: 'generating', label: 'Generating explanation...' },
];

export default function AnalysisLoading({ currentStage }: { currentStage: string }) {
  const currentIndex = STAGES.findIndex(s => s.id === currentStage);
  
  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm max-w-md mx-auto w-full">
      <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center gap-2">
        <Loader2 className="w-5 h-5 animate-spin text-accent" />
        Analyzing Image
      </h3>
      <div className="space-y-3">
        {STAGES.map((stage, i) => {
          const isPast = i < currentIndex;
          const isCurrent = i === currentIndex;
          
          return (
            <div key={stage.id} className={`flex items-center gap-3 ${isPast || isCurrent ? 'text-gray-900' : 'text-gray-400'}`}>
              {isPast ? (
                <CheckCircle2 className="w-5 h-5 text-accent" />
              ) : isCurrent ? (
                <Loader2 className="w-5 h-5 text-accent animate-spin" />
              ) : (
                <Circle className="w-5 h-5 text-gray-300" />
              )}
              <span className={isCurrent ? 'font-medium' : 'text-sm'}>{stage.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}""",
    "src/components/PredictionCard.tsx": """import { AnalysisResult } from '../types';
import SeverityBadge from './SeverityBadge';
import ConfidenceBadge from './ConfidenceBadge';

interface Props {
  result: AnalysisResult;
}

export default function PredictionCard({ result }: Props) {
  if (!result) return null;
  
  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6 flex flex-col gap-4">
      <div>
        <div className="flex justify-between items-start mb-2">
          <h2 className="text-2xl font-bold text-gray-900">{result.plant}</h2>
          {result.isDiseased ? (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
              DISEASED
            </span>
          ) : (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              HEALTHY
            </span>
          )}
        </div>
        {result.isDiseased ? (
          <p className="text-lg text-gray-700">{result.disease}</p>
        ) : (
          <p className="text-lg text-gray-700">No disease detected</p>
        )}
      </div>
      
      {result.isDiseased && result.severity && (
        <div className="grid grid-cols-2 gap-4 py-4 border-y border-gray-100">
          <div>
            <p className="text-sm text-gray-500 mb-1">Severity</p>
            <SeverityBadge level={result.severity} />
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">Affected Area</p>
            <div className="flex items-center gap-2">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-red-500 h-2 rounded-full" 
                  style={{ width: `${(result.affectedAreaPercentage || 0) * 100}%` }}
                />
              </div>
              <span className="text-sm font-medium">
                {((result.affectedAreaPercentage || 0) * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      )}
      
      <div>
        <p className="text-sm text-gray-500 mb-1">Confidence</p>
        <ConfidenceBadge level={result.confidenceLevel} score={result.confidence} />
      </div>
      
      {result.topPredictions && result.topPredictions.length > 0 && (
        <div className="mt-2">
          <details className="text-sm group">
            <summary className="cursor-pointer text-gray-500 hover:text-gray-700 font-medium list-none flex items-center gap-1">
              <span className="group-open:hidden">▶</span>
              <span className="hidden group-open:inline">▼</span>
              Other Predictions
            </summary>
            <ul className="mt-2 space-y-1 text-gray-600 pl-4">
              {result.topPredictions.slice(1).map((p, i) => (
                <li key={i} className="flex justify-between">
                  <span>{p.plant} - {p.disease}</span>
                  <span>{(p.confidence * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </details>
        </div>
      )}
      
      {result.modelInfo && (
        <div className="mt-auto pt-4 text-xs text-gray-400 flex justify-between">
          <span>Model {result.modelInfo.version}</span>
          <span>{result.modelInfo.inferenceTime}ms</span>
        </div>
      )}
    </div>
  );
}""",
    "src/components/ImageViewer.tsx": """import { useState } from 'react';

interface Props {
  images: {
    original: string;
    mask?: string;
    gradcam?: string;
    overlay?: string;
  };
}

export default function ImageViewer({ images }: Props) {
  const tabs = [
    { id: 'original', label: 'Original', url: images.original },
    { id: 'mask', label: 'Disease Mask', url: images.mask },
    { id: 'gradcam', label: 'Grad-CAM', url: images.gradcam },
    { id: 'overlay', label: 'Overlay', url: images.overlay },
  ].filter(t => t.url);
  
  const [activeTab, setActiveTab] = useState(tabs[0]?.id);
  const activeUrl = tabs.find(t => t.id === activeTab)?.url;
  
  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden flex flex-col h-full">
      <div className="flex border-b border-gray-200 bg-gray-50 overflow-x-auto">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors ${
              activeTab === tab.id 
                ? 'text-accent border-b-2 border-accent bg-white' 
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="flex-1 p-4 flex items-center justify-center bg-gray-100 min-h-[300px]">
        {activeUrl ? (
          <img 
            src={activeUrl} 
            alt={activeTab} 
            className="max-w-full max-h-[500px] object-contain rounded shadow-sm"
          />
        ) : (
          <span className="text-gray-400">Image not available</span>
        )}
      </div>
    </div>
  );
}""",
    "src/components/SeverityBadge.tsx": """interface Props {
  level: 'MINIMAL' | 'MILD' | 'MODERATE' | 'SEVERE';
}

const colors = {
  MINIMAL: 'bg-green-100 text-green-800',
  MILD: 'bg-yellow-100 text-yellow-800',
  MODERATE: 'bg-orange-100 text-orange-800',
  SEVERE: 'bg-red-100 text-red-800',
};

export default function SeverityBadge({ level }: Props) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-medium uppercase ${colors[level] || 'bg-gray-100 text-gray-800'}`}>
      {level}
    </span>
  );
}""",
    "src/components/ConfidenceBadge.tsx": """interface Props {
  level: 'HIGH' | 'MEDIUM' | 'LOW';
  score: number;
}

const colors = {
  HIGH: 'bg-green-100 text-green-800',
  MEDIUM: 'bg-amber-100 text-amber-800',
  LOW: 'bg-red-100 text-red-800',
};

export default function ConfidenceBadge({ level, score }: Props) {
  return (
    <div className="flex items-center gap-2">
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-medium uppercase ${colors[level] || 'bg-gray-100 text-gray-800'}`}>
        {level}
      </span>
      <span className="text-sm text-gray-600">{(score * 100).toFixed(1)}%</span>
    </div>
  );
}""",
    "src/components/RejectedResult.tsx": """import { AlertTriangle } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function RejectedResult() {
  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-8 text-center max-w-lg mx-auto mt-12">
      <div className="mx-auto w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mb-4">
        <AlertTriangle className="w-6 h-6 text-red-600" />
      </div>
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Image Not Supported</h2>
      <p className="text-gray-600 mb-6">
        No supported plant leaf detected in this image.<br />
        Please upload a clear image of a supported plant leaf.
      </p>
      <Link to="/" className="inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-accent hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-accent transition-colors">
        Upload Another Image
      </Link>
    </div>
  );
}""",
    "src/components/UncertainResult.tsx": """import { Info } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function UncertainResult() {
  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-8 text-center max-w-lg mx-auto mt-12">
      <div className="mx-auto w-12 h-12 bg-amber-100 rounded-full flex items-center justify-center mb-4">
        <Info className="w-6 h-6 text-amber-600" />
      </div>
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Analysis Uncertain</h2>
      <p className="text-gray-600 mb-6">
        The model could not confidently classify this image.<br />
        Try a clearer image with better lighting.
      </p>
      <Link to="/" className="inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-accent hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-accent transition-colors">
        Upload Another Image
      </Link>
    </div>
  );
}""",
    "src/components/ConfusionMatrix.tsx": """interface Props {
  url: string;
}

export default function ConfusionMatrix({ url }: Props) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6 overflow-x-auto">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Confusion Matrix</h3>
      <div className="min-w-[600px] flex justify-center">
        <img src={url} alt="Confusion Matrix" className="max-w-full h-auto rounded" />
      </div>
    </div>
  );
}""",
    "src/components/MetricsTable.tsx": """import { useState } from 'react';
import { ClassPerformance } from '../types';

interface Props {
  data: ClassPerformance[];
}

export default function MetricsTable({ data }: Props) {
  const [sortConfig, setSortConfig] = useState<{ key: keyof ClassPerformance; direction: 'asc' | 'desc' } | null>(null);

  const sortedData = [...data].sort((a, b) => {
    if (!sortConfig) return 0;
    const { key, direction } = sortConfig;
    if (a[key] < b[key]) return direction === 'asc' ? -1 : 1;
    if (a[key] > b[key]) return direction === 'asc' ? 1 : -1;
    return 0;
  });

  const requestSort = (key: keyof ClassPerformance) => {
    let direction: 'asc' | 'desc' = 'asc';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const getSortIcon = (key: keyof ClassPerformance) => {
    if (!sortConfig || sortConfig.key !== key) return '↕';
    return sortConfig.direction === 'asc' ? '↑' : '↓';
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {['plant', 'disease', 'precision', 'recall', 'f1', 'support'].map((key) => (
                <th
                  key={key}
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => requestSort(key as keyof ClassPerformance)}
                >
                  {key.charAt(0).toUpperCase() + key.slice(1)} <span className="ml-1">{getSortIcon(key as keyof ClassPerformance)}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedData.map((row, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{row.plant}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{row.disease}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{(row.precision * 100).toFixed(1)}%</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{(row.recall * 100).toFixed(1)}%</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{(row.f1 * 100).toFixed(1)}%</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{row.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}""",
    "src/components/PerformanceChart.tsx": """import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Metrics } from '../types';

interface Props {
  metrics: Metrics;
}

export default function PerformanceChart({ metrics }: Props) {
  const data = [
    { name: 'Accuracy', value: metrics.accuracy * 100 },
    { name: 'Precision', value: metrics.precision * 100 },
    { name: 'Recall', value: metrics.recall * 100 },
    { name: 'F1 Score', value: metrics.f1 * 100 },
  ];

  return (
    <div className="h-64 w-full bg-white rounded-lg border border-gray-200 shadow-sm p-4 mt-6">
      <h3 className="text-sm font-medium text-gray-500 mb-4">Overall Metrics (%)</h3>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} stroke="#f3f4f6" />
          <XAxis type="number" domain={[0, 100]} />
          <YAxis dataKey="name" type="category" width={80} tick={{ fontSize: 12 }} />
          <Tooltip cursor={{fill: '#f9fafb'}} formatter={(value: number) => [`${value.toFixed(1)}%`, 'Score']} />
          <Bar dataKey="value" fill="#16a34a" radius={[0, 4, 4, 0]} barSize={24} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}""",
    "src/pages/AnalyzePage.tsx": """import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ImageUploader from '../components/ImageUploader';
import AnalysisLoading from '../components/AnalysisLoading';
import { useAnalysis } from '../hooks/useAnalysis';
import { getSupportedPlants } from '../services/api';

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [supportedPlants, setSupportedPlants] = useState<string[]>([]);
  const { analyze, isAnalyzing, stage, error } = useAnalysis();
  const navigate = useNavigate();

  useEffect(() => {
    getSupportedPlants().then(setSupportedPlants).catch(() => {});
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;
    const id = await analyze(file);
    if (id) {
      navigate(`/result/${id}`);
    }
  };

  return (
    <div className="max-w-3xl mx-auto py-12">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-extrabold text-gray-900 tracking-tight sm:text-5xl mb-4">
          Plant Disease Detection & Visual Analysis
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Upload a clear image of a leaf to instantly identify diseases, assess severity, and see visual explanations.
        </p>
      </div>

      <div className="space-y-8">
        {!isAnalyzing ? (
          <div className="bg-white p-6 sm:p-8 rounded-xl border border-gray-200 shadow-sm">
            <ImageUploader 
              selectedFile={file} 
              onImageSelected={setFile} 
              onClear={() => setFile(null)} 
            />
            
            {error && (
              <div className="mt-4 p-3 bg-red-50 text-red-700 text-sm rounded-md border border-red-200">
                {error}
              </div>
            )}
            
            {file && (
              <div className="mt-6 flex justify-end">
                <button
                  onClick={handleAnalyze}
                  className="px-6 py-3 bg-accent text-white font-medium rounded-lg shadow-sm hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-accent transition-colors w-full sm:w-auto"
                >
                  🔍 Analyze Leaf
                </button>
              </div>
            )}
          </div>
        ) : (
          <AnalysisLoading currentStage={stage} />
        )}

        <div className="text-center">
          <p className="text-sm text-gray-500 mb-2">Supported Plants</p>
          <div className="flex flex-wrap justify-center gap-2">
            {supportedPlants.map(plant => (
              <span key={plant} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                {plant}
              </span>
            ))}
            <span className="inline-flex items-center px-2.5 py-0.5 text-xs text-gray-500">...and more</span>
          </div>
        </div>
      </div>
    </div>
  );
}""",
    "src/pages/ResultPage.tsx": """import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getAnalysis } from '../services/api';
import { AnalysisResult } from '../types';
import ImageViewer from '../components/ImageViewer';
import PredictionCard from '../components/PredictionCard';
import RejectedResult from '../components/RejectedResult';
import UncertainResult from '../components/UncertainResult';
import { ArrowLeft, Loader2 } from 'lucide-react';

export default function ResultPage() {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!id) return;
    getAnalysis(id)
      .then(res => {
        setResult(res);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Failed to load result');
        setLoading(false);
      });
  }, [id]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh]">
        <Loader2 className="w-8 h-8 text-accent animate-spin mb-4" />
        <p className="text-gray-500">Loading result...</p>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="text-center mt-12">
        <p className="text-red-500 mb-4">{error || 'Result not found'}</p>
        <Link to="/" className="text-accent hover:underline flex items-center justify-center gap-2">
          <ArrowLeft className="w-4 h-4" /> Back to Analysis
        </Link>
      </div>
    );
  }

  if (result.status === 'rejected') return <RejectedResult />;
  if (result.status === 'uncertain') return <UncertainResult />;

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6 flex items-center justify-between">
        <Link to="/" className="text-secondary hover:text-primary flex items-center gap-2 text-sm font-medium transition-colors">
          <ArrowLeft className="w-4 h-4" /> Analyze Another Image
        </Link>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        <div className="h-[500px] lg:h-[600px]">
          {result.images && <ImageViewer images={result.images} />}
        </div>
        
        <div className="flex flex-col gap-6">
          <PredictionCard result={result} />
        </div>
      </div>
    </div>
  );
}""",
    "src/pages/PerformancePage.tsx": """import { useEffect, useState } from 'react';
import { getPerformance } from '../services/api';
import { PerformanceData } from '../types';
import MetricsTable from '../components/MetricsTable';
import PerformanceChart from '../components/PerformanceChart';
import ConfusionMatrix from '../components/ConfusionMatrix';
import { Loader2 } from 'lucide-react';

export default function PerformancePage() {
  const [data, setData] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    getPerformance()
      .then(res => {
        setData(res);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Model has not been evaluated yet.');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh]">
        <Loader2 className="w-8 h-8 text-accent animate-spin mb-4" />
        <p className="text-gray-500">Loading performance data...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center mt-12 bg-white rounded-lg border border-gray-200 p-8 shadow-sm max-w-lg mx-auto">
        <p className="text-gray-600 font-medium">{error}</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Performance</h1>
        <p className="text-gray-500">Evaluation metrics and inference statistics.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6 col-span-1 md:col-span-2">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Model Overview</h2>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-6">
            <div>
              <dt className="text-sm font-medium text-gray-500">Architecture</dt>
              <dd className="mt-1 text-sm text-gray-900">{data.modelInfo.architecture}</dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-500">Version</dt>
              <dd className="mt-1 text-sm text-gray-900">{data.modelInfo.version}</dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-500">Dataset</dt>
              <dd className="mt-1 text-sm text-gray-900">{data.modelInfo.dataset}</dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-500">Classes</dt>
              <dd className="mt-1 text-sm text-gray-900">{data.modelInfo.classes}</dd>
            </div>
          </dl>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Inference Speed</h2>
          <dl className="space-y-4">
            <div className="flex justify-between items-center pb-3 border-b border-gray-100">
              <dt className="text-sm font-medium text-gray-500">Mean Latency</dt>
              <dd className="text-sm font-semibold text-gray-900">{data.inferencePerformance.meanLatency}ms</dd>
            </div>
            <div className="flex justify-between items-center pb-3 border-b border-gray-100">
              <dt className="text-sm font-medium text-gray-500">P95 Latency</dt>
              <dd className="text-sm font-semibold text-gray-900">{data.inferencePerformance.p95Latency}ms</dd>
            </div>
            <div className="flex justify-between items-center">
              <dt className="text-sm font-medium text-gray-500">Model Size</dt>
              <dd className="text-sm font-semibold text-gray-900">{data.modelInfo.size}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-6">Overall Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Accuracy', value: data.overallMetrics.accuracy },
            { label: 'Precision', value: data.overallMetrics.precision },
            { label: 'Recall', value: data.overallMetrics.recall },
            { label: 'F1 Score', value: data.overallMetrics.f1 },
          ].map(m => (
            <div key={m.label} className="bg-gray-50 p-4 rounded-lg border border-gray-100">
              <p className="text-sm font-medium text-gray-500 mb-1">{m.label}</p>
              <p className="text-2xl font-bold text-gray-900">{(m.value * 100).toFixed(1)}%</p>
            </div>
          ))}
        </div>
        <PerformanceChart metrics={data.overallMetrics} />
      </div>

      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Per-Class Performance</h2>
        <MetricsTable data={data.perClassPerformance} />
      </div>

      {data.confusionMatrixUrl && (
        <ConfusionMatrix url={data.confusionMatrixUrl} />
      )}

      {data.limitations && data.limitations.length > 0 && (
        <div className="bg-orange-50 rounded-lg border border-orange-200 p-6">
          <h2 className="text-lg font-medium text-orange-900 mb-4">Known Limitations</h2>
          <ul className="list-disc pl-5 space-y-2 text-sm text-orange-800">
            {data.limitations.map((lim, idx) => (
              <li key={idx}>{lim}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}"""
}

os.makedirs(base_dir, exist_ok=True)

for rel_path, content in files.items():
    full_path = os.path.join(base_dir, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

print("SUCCESS: All frontend files created.")
