import type { JobResults } from './types';

const microscope = '/demo/DSCN3052.JPG';
const talcOverlay = '/demo/talc_overlay.png';
const coarseOverlay = '/demo/coarse_overlay.png';
const sulfideCvOverlay = '/demo/sulfide_cv_overlay.png';
const sulfideSamOverlay = '/demo/sulfide_sam_overlay.png';

export const demoResults: JobResults = {
  job_id: 'DEMO-0001',
  status: 'completed',
  demo: true,
  items: [
    {
      image_id: 'sample-01',
      filename: 'DSCN3052.JPG',
      status: 'completed',
      classification: { code: 'ordinary', label: 'Рядовая', confidence: 0.91 },
      talc: { percent: 7.4, coarse_percent: 13.8, refined_percent: 7.4, confidence: 0.87 },
      sulfide: { probability_ordinary: 0.91, probability_difficult: 0.09, confidence: 0.91 },
      sulfide_segmentation: {
        cv: { percent: 11.3, pixel_count: 142000, component_count: 9 },
        sam: { percent: 13.8, pixel_count: 173000, component_count: 7 },
        selected: 'sam',
        sam_error: null,
      },
      timings: {
        preprocessing: 0.4,
        segmentation: 8.7,
        cv_refinement: 1.2,
        sulfide_segmentation: 0.8,
        sulfide: 2.3,
        total: 12.9,
      },
      artifacts: {
        original: microscope,
        talc_mask: talcOverlay,
        coarse_mask: coarseOverlay,
        sulfide_cv_overlay: sulfideCvOverlay,
        sulfide_sam_overlay: sulfideSamOverlay,
        overlay: talcOverlay,
      },
    },
    {
      image_id: 'sample-02',
      filename: 'DSCN3052.JPG · альтернативный порог',
      status: 'completed',
      classification: { code: 'talc_bearing', label: 'Оталькованная', confidence: null },
      talc: { percent: 12.6, coarse_percent: 18.2, refined_percent: 12.6, confidence: 0.84 },
      sulfide: null,
      sulfide_segmentation: {
        cv: { percent: 8.1, pixel_count: 99000, component_count: 5 },
        sam: { percent: 9.4, pixel_count: 115000, component_count: 4 },
        selected: 'sam',
        sam_error: null,
      },
      timings: { preprocessing: 0.5, segmentation: 9.4, cv_refinement: 1.4, sulfide_segmentation: 0.7, sulfide: 2.1, total: 13.7 },
      artifacts: {
        original: microscope,
        talc_mask: talcOverlay,
        coarse_mask: coarseOverlay,
        sulfide_cv_overlay: sulfideCvOverlay,
        sulfide_sam_overlay: sulfideSamOverlay,
      },
    },
  ],
};
