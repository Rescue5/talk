import type { JobResults } from './types';

const microscope = `data:image/svg+xml,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">
<defs>
<filter id="n"><feTurbulence baseFrequency=".015" numOctaves="4" seed="12"/><feColorMatrix values=".2 0 0 0 .12 0 .24 0 0 .14 0 0 .18 0 .13 0 0 0 .85 0"/></filter>
<radialGradient id="g"><stop stop-color="#7a7466"/><stop offset="1" stop-color="#272b26"/></radialGradient>
</defs>
<rect width="1400" height="900" fill="url(#g)"/><rect width="1400" height="900" filter="url(#n)" opacity=".72"/>
<g fill="#e7e1cd" opacity=".86"><path d="M805 234l88 49 36 97-61 86-112-16-54-106z"/><path d="M537 572l70-43 102 37 14 79-68 72-104-37z"/></g>
<g fill="#171a17"><circle cx="770" cy="316" r="17"/><circle cx="834" cy="341" r="11"/><circle cx="863" cy="400" r="15"/><circle cx="598" cy="612" r="12"/><circle cx="650" cy="656" r="18"/></g>
</svg>`)}`

const talcOverlay = `data:image/svg+xml,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900"><g fill="none" stroke="#c95f3c" stroke-width="10"><path d="M746 262l104-13 87 69 5 105-72 74-120-20-73-98 25-82z"/><path d="M523 543l79-42 112 25 44 104-69 105-121-19-78-96z"/></g></svg>`)}`

const coarseOverlay = `data:image/svg+xml,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900"><path d="M675 208c165-56 303 24 332 168 26 128-55 191-196 162-68-13-125-32-171-100-52-75-55-180 35-230zM475 482c135-59 280 9 319 132 33 102-38 186-142 173-145-18-244-82-220-201 9-45 12-82 43-104z" fill="#c95f3c" fill-opacity=".14" stroke="#d57a5b" stroke-width="6" stroke-dasharray="18 11"/></svg>`)}`

const sulfideCvOverlay = `data:image/svg+xml,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900"><g fill="#ffb226" fill-opacity=".68"><path d="M801 232l92 51 38 98-62 88-114-17-54-109z"/><path d="M539 572l69-43 99 37 15 78-67 70-102-37z"/></g></svg>`)}`

const sulfideSamOverlay = `data:image/svg+xml,${encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900"><g fill="#2490ff" fill-opacity=".58"><path d="M789 220l114 49 49 109-70 105-140-19-66-122z"/><path d="M525 559l83-49 116 42 21 93-78 88-128-42z"/></g></svg>`)}`

export const demoResults: JobResults = {
  job_id: 'DEMO-0001',
  status: 'completed',
  demo: true,
  items: [
    {
      image_id: 'sample-01',
      filename: 'sample_ore_01.jpg',
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
      filename: 'sample_ore_02.jpg',
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
