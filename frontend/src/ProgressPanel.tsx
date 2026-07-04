import { Check, CircleDot, LoaderCircle, TriangleAlert } from 'lucide-react';
import { errorMessage, type Job } from './types';

const STAGES = [
  ['upload', 'Загрузка'],
  ['talc_segmentation', 'Сегментация'],
  ['cv_refinement', 'CV-уточнение'],
  ['sulfide_segmentation', 'Сульфиды'],
  ['sulfide_classification', 'Классификация'],
  ['export', 'Экспорт'],
] as const;

export function ProgressPanel({
  job,
  files = [],
  uploadPercent = 100,
}: {
  job: Job;
  files?: File[];
  uploadPercent?: number;
}) {
  const progress = Math.max(0, Math.min(100, job.progress?.percent ?? uploadPercent));
  const active = job.progress?.stage?.toLowerCase() ?? 'upload';
  const activeIndex = Math.max(
    0,
    STAGES.findIndex(([key]) => active.includes(key)),
  );
  const isError = ['failed', 'model_unavailable'].includes(job.status);

  return (
    <section className={`progress-panel ${isError ? 'is-error' : ''}`} aria-live="polite">
      <div className="progress-heading">
        <div>
          <span className="eyebrow">Задача {job.id.slice(0, 8)}</span>
          <h2>{isError ? 'Обработка остановлена' : job.progress?.message || 'Анализ изображений'}</h2>
        </div>
        <strong>{Math.round(progress)}%</strong>
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress)}
      >
        <span style={{ width: `${progress}%` }} />
      </div>
      <div className="stage-row">
        {STAGES.map(([key, label], index) => {
          const done = progress === 100 || index < activeIndex;
          const current = index === activeIndex && progress < 100;
          return (
            <div className={done ? 'done' : current ? 'current' : ''} key={key}>
              {isError && current ? (
                <TriangleAlert size={15} />
              ) : done ? (
                <Check size={15} />
              ) : current ? (
                <LoaderCircle className="spin" size={15} />
              ) : (
                <CircleDot size={15} />
              )}
              <span>{label}</span>
            </div>
          );
        })}
      </div>
      {job.progress && (
        <p className="progress-count">
          Готово {job.progress.completed_images} из {job.progress.total_images} изображений
        </p>
      )}
      {files.length > 0 && (
        <div className="progress-files" aria-label="Состояние изображений">
          {files.slice(0, 5).map((file, index) => {
            const complete = index < (job.progress?.completed_images ?? 0);
            const activeFile = index === (job.progress?.completed_images ?? 0) && job.status === 'running';
            return (
              <div key={`${file.name}-${file.size}`} className={complete ? 'done' : activeFile ? 'active' : ''}>
                <i>{complete ? <Check size={11} /> : String(index + 1).padStart(2, '0')}</i>
                <span>{file.name}</span>
                <small>{complete ? 'готово' : activeFile ? 'обработка' : 'в очереди'}</small>
              </div>
            );
          })}
          {files.length > 5 && <small className="more-files">+ ещё {files.length - 5}</small>}
        </div>
      )}
      {job.error && (
        <p className="error-copy">
          {errorMessage(job.error, 'Неизвестная ошибка обработки')}
        </p>
      )}
    </section>
  );
}
