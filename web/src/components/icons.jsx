// Линейные иконки одним файлом.
//
// Рисуем сами, а не тянем библиотеку: нужно полтора десятка штук, а любой
// готовый набор — это лишняя зависимость в образе и лишние сотни килобайт в
// бандле ради того же результата. Все иконки в одной сетке 24×24 с одинаковой
// толщиной штриха, иначе в ряду они начинают «прыгать».

const base = {
  width: 22,
  height: 22,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.7,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
}

export const IconOverview = (p) => (
  <svg {...base} {...p}>
    <rect x="3" y="3" width="7.5" height="7.5" rx="1.5" />
    <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5" />
    <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5" />
    <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5" />
  </svg>
)

export const IconMap = (p) => (
  <svg {...base} {...p}>
    <path d="M3 6.5 9 4l6 2.5L21 4v13.5L15 20l-6-2.5L3 20z" />
    <path d="M9 4v13.5M15 6.5V20" />
  </svg>
)

export const IconFields = (p) => (
  <svg {...base} {...p}>
    <path d="M12 3 21 8l-9 5-9-5z" />
    <path d="M3 12.5 12 17.5 21 12.5" />
    <path d="M3 16.5 12 21.5 21 16.5" />
  </svg>
)

export const IconAnalytics = (p) => (
  <svg {...base} {...p}>
    <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
  </svg>
)

export const IconReports = (p) => (
  <svg {...base} {...p}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5M9 13h6M9 17h4" />
  </svg>
)

export const IconPin = (p) => (
  <svg {...base} {...p}>
    <path d="M12 21s7-6.2 7-11a7 7 0 1 0-14 0c0 4.8 7 11 7 11z" />
    <circle cx="12" cy="10" r="2.5" />
  </svg>
)

export const IconCalendar = (p) => (
  <svg {...base} {...p}>
    <rect x="3" y="5" width="18" height="16" rx="2.5" />
    <path d="M3 10h18M8 3v4M16 3v4" />
    <rect x="6.5" y="13" width="4" height="4" rx="1" />
  </svg>
)

export const IconBell = (p) => (
  <svg {...base} {...p}>
    <path d="M18 9a6 6 0 1 0-12 0c0 5-2 6-2 6h16s-2-1-2-6z" />
    <path d="M10.5 20a2 2 0 0 0 3 0" />
  </svg>
)

export const IconChevron = (p) => (
  <svg {...base} width="16" height="16" {...p}>
    <path d="M6 9.5 12 15.5 18 9.5" />
  </svg>
)

export const IconArrow = (p) => (
  <svg {...base} {...p}>
    <path d="M4 12h15M13.5 6.5 19.5 12l-6 5.5" />
  </svg>
)

export const IconDownload = (p) => (
  <svg {...base} {...p}>
    <path d="M12 3v12M7.5 10.5 12 15l4.5-4.5M4 20h16" />
  </svg>
)

export const IconTrash = (p) => (
  <svg {...base} width="17" height="17" {...p}>
    <path d="M4 7h16M10 4h4M6 7l1 13h10l1-13M10 11v6M14 11v6" />
  </svg>
)

export const IconPencil = (p) => (
  <svg {...base} width="17" height="17" {...p}>
    <path d="M4 20h4L20 8l-4-4L4 16z" />
  </svg>
)

export const IconUser = (p) => (
  <svg {...base} width="18" height="18" {...p}>
    <circle cx="12" cy="8.5" r="3.6" />
    <path d="M4.8 20c0-3.6 3.2-5.6 7.2-5.6s7.2 2 7.2 5.6" />
  </svg>
)

/* --- крупные глифы карточек показателей: под макет, каждый своим цветом --- */

const glyph = {
  width: 58,
  height: 58,
  viewBox: '0 0 48 48',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.9,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
}

// Колос: участки и всё, что про сами поля.
export const GlyphSpike = (p) => (
  <svg {...glyph} {...p}>
    <path d="M9 43 27 25" />
    <path d="M27 25c-2.6-2.6-2.6-6.8 0-9.4 2.6 2.6 2.6 6.8 0 9.4z" />
    <path d="M22 30c-2.6-2.6-6.8-2.6-9.4 0 2.6 2.6 6.8 2.6 9.4 0z" />
    <path d="M27 15.6c-2.6-2.6-2.6-6.8 0-9.4 2.6 2.6 2.6 6.8 0 9.4z" />
    <path d="M17 35c-2.6-2.6-6.8-2.6-9.4 0 2.6 2.6 6.8 2.6 9.4 0z" />
    <path d="M32 20c2.6-2.6 6.8-2.6 9.4 0-2.6 2.6-6.8 2.6-9.4 0z" transform="translate(-9 6)" />
  </svg>
)

// График с пунктиром: найденные аномалии.
export const GlyphChart = (p) => (
  <svg {...glyph} {...p}>
    <path d="M8 6v34h34" />
    <path d="M13 32l8-9 6 5 11-14" />
    <path d="M13 14h.01M20 12h.01M27 10h.01M34 8h.01M41 20h.01M34 22h.01" strokeWidth="2.6" />
  </svg>
)

// Облако с дождём: свежесть данных и погода.
export const GlyphCloud = (p) => (
  <svg {...glyph} {...p}>
    <path d="M15 27a7 7 0 0 1 .6-13.9 10 10 0 0 1 19 3 6 6 0 0 1-1.6 11H15z" />
    <path d="M16 33l-2 6M24 33l-2 6M32 33l-2 6" />
  </svg>
)

// Знак сервиса: спутниковая тарелка над всходом, в кольце.
//
// Смысл ровно тот же, что у продукта: взгляд из космоса на то, что растёт.
// Рисуем в двух цветах — белая техника, зелёное растение, — чтобы знак читался
// и на тёмной колонке, и уменьшенным до значка в 22 пикселя.
export const Logo = ({ size = 62 }) => (
  <svg width={size} height={size} viewBox="0 0 64 64" fill="none" aria-hidden="true">
    <circle cx="32" cy="32" r="29" stroke="#3a3a3a" strokeWidth="1.8" />

    {/* тарелка и луч */}
    <path
      d="M23.5 27.5a9 9 0 0 1 12.7-12.7z"
      stroke="#f0f0f0"
      strokeWidth="2"
      strokeLinejoin="round"
    />
    <path d="M29 22l6.5 6.5" stroke="#f0f0f0" strokeWidth="2" strokeLinecap="round" />
    <path
      d="M41 14.5l6.5 6.5M44.5 12l6 6M38 18.5l6.5 6.5"
      stroke="#f0f0f0"
      strokeWidth="1.7"
      strokeLinecap="round"
    />

    {/* всход */}
    <path d="M32 50V36" stroke="#4e9b36" strokeWidth="2.4" strokeLinecap="round" />
    <path
      d="M32 38c-5 0-9-3.6-9.6-8.4 5-.6 9 2.6 9.6 8.4z"
      fill="#4e9b36"
      opacity="0.95"
    />
    <path
      d="M32 43c4.4 0 8-3.2 8.5-7.5-4.4-.5-8 2.4-8.5 7.5z"
      fill="#6aa84f"
    />
  </svg>
)
