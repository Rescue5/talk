import { talcPercent, totalSeconds, type ResultItem } from './types';

const valueOf = (value: unknown, fallback = 0) =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

export function ThresholdChart({ item, threshold }: { item: ResultItem; threshold: number }) {
  const value = talcPercent(item);
  const scale = Math.max(20, value, threshold) * 1.15;
  const valueX = Math.min(100, (value / scale) * 100);
  const thresholdX = Math.min(100, (threshold / scale) * 100);
  return (
    <figure className="mini-chart" aria-label={`Тальк ${value.toFixed(1)}%, порог ${threshold}%`}>
      <figcaption>
        <span>Порог классификации</span>
        <strong>{value.toFixed(1)}%</strong>
      </figcaption>
      <svg viewBox="0 0 400 56" role="img">
        <title>Доля талька относительно порога классификации</title>
        <rect x="0" y="18" width="400" height="12" rx="6" className="chart-track" />
        <rect x="0" y="18" width={valueX * 4} height="12" rx="6" className="chart-accent" />
        <line x1={thresholdX * 4} x2={thresholdX * 4} y1="7" y2="41" className="threshold-line" />
        <text x={Math.min(374, thresholdX * 4 + 6)} y="52" className="chart-label">
          {threshold}%
        </text>
      </svg>
    </figure>
  );
}

export function CompositionChart({ item }: { item: ResultItem }) {
  const talc = talcPercent(item);
  const sulfide = valueOf(item.sulfide?.percent);
  const rest = Math.max(0, 100 - talc - sulfide);
  const segments = [
    { label: 'Тальк', value: talc, className: 'talc' },
    { label: 'Сульфиды', value: sulfide, className: 'sulfide' },
    { label: 'Прочее', value: rest, className: 'rest' },
  ];
  let offset = 0;
  return (
    <figure className="mini-chart" aria-label="Состав изображения">
      <figcaption>
        <span>Состав</span>
        <strong>{(talc + sulfide).toFixed(1)}%</strong>
      </figcaption>
      <svg viewBox="0 0 400 58" role="img">
        <title>Доли талька, сульфидов и остальных областей</title>
        {segments.map((segment) => {
          const current = offset;
          offset += segment.value;
          return (
            <rect
              key={segment.label}
              x={current * 4}
              y="10"
              width={Math.max(0, segment.value * 4)}
              height="15"
              className={`composition-${segment.className}`}
            />
          );
        })}
        {segments.map((segment, index) => (
          <g key={segment.label} transform={`translate(${index * 128} 45)`}>
            <circle r="4" className={`composition-${segment.className}`} />
            <text x="9" y="4" className="chart-label">
              {segment.label} {segment.value.toFixed(1)}%
            </text>
          </g>
        ))}
      </svg>
    </figure>
  );
}

export function TimingChart({ item }: { item: ResultItem }) {
  const entries = Object.entries(item.timings ?? {})
    .filter(([key, value]) => key !== 'total' && key !== 'pipeline_total' && typeof value === 'number')
    .map(([key, value]) => ({ key, value: value as number }));
  const max = Math.max(...entries.map((entry) => entry.value), 1);
  const labels: Record<string, string> = {
    preprocessing: 'Подготовка',
    segmentation: 'Сегментация',
    cv_refinement: 'CV',
    sulfide: 'Сульфиды',
  };
  return (
    <figure className="timing-chart">
      <figcaption>
        <span>Время этапов</span>
        <strong>{totalSeconds(item).toFixed(1)} c</strong>
      </figcaption>
      <div role="img" aria-label="Длительность этапов обработки">
        {entries.map(({ key, value }) => (
          <div className="timing-row" key={key}>
            <span>{labels[key] ?? key}</span>
            <i>
              <b style={{ width: `${(value / max) * 100}%` }} />
            </i>
            <em>{value.toFixed(1)} c</em>
          </div>
        ))}
      </div>
    </figure>
  );
}
