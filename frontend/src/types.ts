export type JobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'partial_failed'
  | 'failed'
  | 'model_unavailable';

export type JobSettings = {
  mode: 'overlap' | 'no_overlap';
  talc_threshold_percent: number;
  sulfide_threshold: number;
  segmentation_threshold: number;
  cv_threshold: number;
};

export type JobProgress = {
  percent: number;
  stage: string;
  completed_images: number;
  total_images: number;
  message?: string | null;
};

export type ImageProgress = {
  percent: number;
  stage: string;
  message?: string | null;
};

export type JobImage = {
  image_id: string;
  filename: string;
  status: string;
  settings: JobSettings;
  progress: ImageProgress;
  created_at?: string;
  updated_at?: string;
};

export type ApiError = string | { code?: string; message?: string; [key: string]: unknown } | null;

export type SulfideMaskType = 'cv' | 'sam';

export type SulfideMaskStats = {
  pixel_count?: number;
  percent?: number;
  component_count?: number;
  [key: string]: unknown;
};

export type SulfideSegmentation = {
  cv?: SulfideMaskStats | null;
  sam?: SulfideMaskStats | null;
  selected?: SulfideMaskType;
  sam_error?: ApiError;
  [key: string]: unknown;
};

export type Job = {
  id: string;
  status: JobStatus;
  demo?: boolean;
  progress?: JobProgress;
  images?: JobImage[];
  settings: JobSettings;
  created_at?: string;
  updated_at?: string;
  error?: ApiError;
};

export type ArtifactSet = {
  original?: string;
  overlay?: string;
  talc_mask?: string;
  sulfide_mask?: string;
  sulfide_cv_mask?: string;
  sulfide_sam_mask?: string;
  sulfide_cv_overlay?: string;
  sulfide_sam_overlay?: string;
  coarse_mask?: string;
  segmentation_mask?: string;
  refined_talc_mask?: string;
  coarse_overlay?: string;
  talc_overlay?: string;
  [key: string]: string | undefined;
};

export type Metric = {
  talc_percent?: number;
  threshold_percent?: number;
  percent?: number;
  area_pixels?: number;
  confidence?: number | null;
  coarse_percent?: number;
  refined_percent?: number;
  [key: string]: unknown;
};

export type TimingSet = {
  pipeline_total?: number;
  total?: number;
  segmentation?: number;
  cv_refinement?: number;
  sulfide_segmentation?: number;
  sulfide?: number;
  preprocessing?: number;
  [key: string]: number | undefined;
};

export type ResultItem = {
  image_id: string;
  filename: string;
  status: string;
  classification: {
    code: 'talc_bearing' | 'ordinary' | 'difficult' | null;
    label?: string;
    label_ru?: string;
    confidence?: number | null;
  } | null;
  talc: Metric | null;
  sulfide: (Metric & {
    probability_ordinary?: number;
    probability_difficult?: number;
  }) | null;
  sulfide_segmentation?: SulfideSegmentation | null;
  timings?: TimingSet;
  artifacts: ArtifactSet;
  error?: ApiError;
  settings?: JobSettings;
  progress?: ImageProgress;
};

export type JobResults = {
  job_id: string;
  status: JobStatus;
  demo?: boolean;
  items: ResultItem[];
};

export type HistoryItem = {
  id: string;
  job_id: string;
  image_id: string;
  filename: string;
  status: 'completed' | 'failed' | 'model_unavailable';
  demo: boolean;
  classification: ResultItem['classification'];
  talc: Metric | null;
  sulfide: ResultItem['sulfide'];
  sulfide_segmentation?: SulfideSegmentation | null;
  artifacts: ArtifactSet;
  settings: JobSettings;
  created_at: string;
  updated_at: string;
  error?: ApiError;
};

export function errorMessage(error: ApiError | undefined, fallback: string): string {
  if (!error) return fallback;
  if (typeof error === 'string') return error;
  return error.message || error.code || fallback;
}

export function talcPercent(item: ResultItem): number {
  return Number(item.talc?.talc_percent ?? item.talc?.percent ?? 0);
}

export function sulfideStats(
  item: ResultItem,
  preferred: SulfideMaskType = 'sam',
): SulfideMaskStats | null {
  const segmentation = item.sulfide_segmentation;
  if (!segmentation) return null;
  const selected = segmentation.selected === 'sam' ? 'sam' : 'cv';
  return segmentation[preferred] ?? segmentation[selected] ?? segmentation.cv ?? null;
}

export function totalSeconds(item: ResultItem): number {
  return Number(item.timings?.pipeline_total ?? item.timings?.total ?? 0);
}
