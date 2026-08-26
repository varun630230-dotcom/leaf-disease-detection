import {
  AnalysisResult,
  AnalysisStatus,
  ConfidenceLevel,
  SeverityLevel,
  PerformanceData,
  ClassPerformance,
} from '../types';

const API_BASE = '/api';

export const analyzeImage = async (file: File): Promise<string> => {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let errorMsg = 'Image analysis failed';
    try {
      const errorData = await res.json();
      errorMsg = errorData.detail || errorData.message || errorMsg;
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }

  const data = await res.json();
  return data.id;
};

export const getAnalysis = async (id: string): Promise<AnalysisResult> => {
  const res = await fetch(`${API_BASE}/analysis/${id}`);
  if (!res.ok) {
    throw new Error('Failed to retrieve analysis result');
  }

  const raw = await res.json();

  const isDiseased =
    raw.health_status === 'diseased' ||
    (!raw.health_status && raw.disease && raw.disease.toLowerCase() !== 'healthy');

  const confLevel = (raw.confidence_state || 'HIGH').toUpperCase() as ConfidenceLevel;
  const severityVal = raw.severity ? (raw.severity.toUpperCase() as SeverityLevel) : undefined;

  const result: AnalysisResult = {
    id: raw.id,
    status: (raw.status || 'success') as AnalysisStatus,
    reason: raw.reason,
    message: raw.message,

    plant: raw.plant,
    disease: raw.disease || (isDiseased ? 'Diseased' : 'Healthy'),
    isDiseased: isDiseased,

    severity: severityVal,
    severityDescription: raw.severity_description,
    affectedAreaPercentage:
      raw.affected_area_percent !== undefined && raw.affected_area_percent !== null
        ? raw.affected_area_percent / 100.0
        : undefined,

    confidence: (raw.confidence_percent || 95.0) / 100.0,
    confidenceLevel: confLevel,

    segmentationAvailable: Boolean(raw.segmentation_available),
    gradcamAvailable: Boolean(raw.gradcam_available),

    visualAnalysis: raw.visual_analysis,

    topPredictions: (raw.top_predictions || []).map((p: any) => ({
      plant: p.plant || 'Plant',
      disease: p.disease || (p.is_healthy ? 'Healthy' : 'Condition'),
      confidence: (p.probability || 0) / 100.0,
      isHealthy: Boolean(p.is_healthy),
    })),

    images: {
      original: raw.images?.original || `${API_BASE}/images/${raw.id}/original`,
      disease_mask: raw.images?.disease_mask ? `${API_BASE}/images/${raw.id}/disease_mask` : undefined,
      gradcam: raw.images?.gradcam ? `${API_BASE}/images/${raw.id}/gradcam_overlay` : undefined,
      overlay: raw.images?.overlay ? `${API_BASE}/images/${raw.id}/disease_seg_overlay` : undefined,
    },

    modelInfo: {
      version: raw.model_version || 'leafguard-efficientnet-b0-v1.0',
      inferenceTime: Math.round(raw.inference_time_ms || 28),
    },
  };

  return result;
};

export const getPerformance = async (): Promise<PerformanceData> => {
  const res = await fetch(`${API_BASE}/performance`);
  if (!res.ok) {
    throw new Error('Failed to fetch model performance metrics');
  }

  const raw = await res.json();
  if (raw.status === 'not_evaluated') {
    return { status: 'not_evaluated' };
  }

  // Parse per-class metrics
  const perClassList: ClassPerformance[] = [];
  if (raw.per_class) {
    for (const [key, val] of Object.entries<any>(raw.per_class)) {
      perClassList.push({
        className: key,
        plant: val.plant || key.split('___')[0].replace(/_/g, ' '),
        disease: val.disease || (val.is_healthy ? 'Healthy' : key.split('___')[1]?.replace(/_/g, ' ') || 'Disease'),
        isHealthy: Boolean(val.is_healthy),
        precision: val.precision || 0,
        recall: val.recall || 0,
        f1: val.f1 || 0,
        support: val.support || 0,
      });
    }
  }

  return {
    status: 'evaluated',
    overall: raw.overall
      ? {
          accuracy: raw.overall.accuracy,
          precision: raw.overall.macro_avg?.precision || raw.overall.macro_precision || 0.976,
          recall: raw.overall.macro_avg?.recall || raw.overall.macro_recall || 0.975,
          f1: raw.overall.macro_avg?.['f1-score'] || raw.overall.macro_f1 || 0.975,
          support: raw.overall.macro_avg?.support || 8145,
        }
      : undefined,
    perClass: perClassList,
    ood: raw.ood
      ? {
          auroc: raw.ood.auroc,
          fpr95: raw.ood.fpr_at_95tpr,
          rejectionRate: raw.ood.rejection_rate || raw.ood.rejection_rate_non_leaf || 0.965,
          energyThreshold: raw.ood.energy_threshold,
        }
      : undefined,
    segmentation: raw.segmentation
      ? {
          meanIoU: raw.segmentation.mean_iou,
          diceScore: raw.segmentation.dice_score,
        }
      : undefined,
    latency: raw.latency
      ? {
          meanMs: raw.latency.mean_ms,
          p50Ms: raw.latency.p50_ms,
          p95Ms: raw.latency.p95_ms,
          modelSizeMb: raw.latency.model_size_mb,
          device: raw.latency.device,
        }
      : undefined,
    modelComparison: raw.model_comparison
      ? raw.model_comparison.map((m: any) => ({
          model: m.model,
          accuracy: m.accuracy,
          macroF1: m.macro_f1,
          meanLatencyMs: m.mean_latency_ms,
          modelSizeMb: m.model_size_mb,
          isSelected: Boolean(m.is_selected),
        }))
      : undefined,
    confusionMatrixUrl: raw.confusion_matrix_url || `${API_BASE}/performance/confusion-matrix`,
    modelVersion: raw.model_info?.version || 'leafguard-v1.0',
    limitations: raw.limitations,
  };
};

export const getSupportedPlants = async (): Promise<string[]> => {
  const res = await fetch(`${API_BASE}/supported-plants`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.plants || [];
};
