/** TypeScript Type Definitions for LeafGuard AI. */

export type AnalysisStatus = 'processing' | 'success' | 'rejected' | 'uncertain' | 'error';
export type SeverityLevel = 'MINIMAL' | 'MILD' | 'MODERATE' | 'SEVERE';
export type ConfidenceLevel = 'HIGH' | 'MEDIUM' | 'LOW';

export interface TopPrediction {
  plant: string;
  disease: string;
  confidence: number;
  isHealthy: boolean;
}

export interface AnalysisImages {
  original: string;
  disease_mask?: string;
  gradcam?: string;
  overlay?: string;
}

export interface ModelInfo {
  version: string;
  inferenceTime: number;
  isMock?: boolean;
}

export interface AnalysisResult {
  id: string;
  status: AnalysisStatus;
  reason?: string;
  message?: string;

  // Plant & Condition
  plant?: string;
  disease?: string;
  isDiseased?: boolean;

  // Severity & Area
  severity?: SeverityLevel;
  severityDescription?: string;
  affectedAreaPercentage?: number;

  // Confidence
  confidence: number;
  confidenceLevel: ConfidenceLevel;

  // Feature Availability
  segmentationAvailable: boolean;
  gradcamAvailable: boolean;

  // Concise Explanation
  visualAnalysis?: string;

  // Top Predictions
  topPredictions?: TopPrediction[];

  // Visual Images
  images?: AnalysisImages;

  // Model Metadata
  modelInfo?: ModelInfo;
}

// Performance & Benchmarks
export interface OverallMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  support?: number;
}

export interface ClassPerformance {
  className: string;
  plant: string;
  disease: string;
  isHealthy: boolean;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface OODMetrics {
  auroc: number;
  fpr95: number;
  rejectionRate: number;
  energyThreshold?: number;
}

export interface SegmentationMetrics {
  meanIoU: number;
  diceScore: number;
}

export interface LatencyMetrics {
  meanMs: number;
  p50Ms: number;
  p95Ms: number;
  modelSizeMb: number;
  device?: string;
}

export interface ModelComparisonItem {
  model: string;
  accuracy: number;
  macroF1: number;
  meanLatencyMs: number;
  modelSizeMb: number;
  isSelected: boolean;
}

export interface PerformanceData {
  status: 'evaluated' | 'not_evaluated';
  overall?: OverallMetrics;
  perClass?: ClassPerformance[];
  ood?: OODMetrics;
  segmentation?: SegmentationMetrics;
  latency?: LatencyMetrics;
  modelComparison?: ModelComparisonItem[];
  confusionMatrixUrl?: string;
  modelVersion?: string;
  limitations?: string[];
}
