import { describe, expect, it } from 'vitest';
import { errorMessage, sulfideStats, talcPercent, totalSeconds, type ResultItem } from './types';

const result = {
  image_id: 'image',
  filename: 'image.png',
  status: 'completed',
  classification: { code: 'ordinary' as const, label_ru: 'рядовая руда' },
  talc: { talc_percent: 7.25 },
  sulfide: { probability_ordinary: 0.82 },
  sulfide_segmentation: {
    cv: { percent: 4.2, component_count: 3 },
    sam: { percent: 5.1, component_count: 2 },
    selected: 'sam',
  },
  timings: { pipeline_total: 4.2 },
  artifacts: { original: '/api/original' },
} satisfies ResultItem;

describe('backend result adapters', () => {
  it('reads canonical backend metric fields', () => {
    expect(talcPercent(result)).toBe(7.25);
    expect(totalSeconds(result)).toBe(4.2);
    expect(sulfideStats(result, 'sam')?.percent).toBe(5.1);
    expect(sulfideStats({ ...result, sulfide_segmentation: { cv: { percent: 4.2 } } }, 'sam')?.percent)
      .toBe(4.2);
  });

  it('formats structured backend errors', () => {
    expect(errorMessage({ code: 'model_unavailable', message: 'Checkpoint missing' }, 'Fallback'))
      .toBe('Checkpoint missing');
    expect(errorMessage(null, 'Fallback')).toBe('Fallback');
  });
});
