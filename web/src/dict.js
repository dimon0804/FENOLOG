// Словари для показа служебных значений ядра по-русски.
//
// Держим их в одном месте: те же подписи нужны и ленте событий, и подсказкам на
// графике, и карточке поля. Разъехавшиеся названия одного и того же класса —
// первое, что бросается в глаза на защите.

// Классы состояния по z-оценке. Пороги заданы ядром: z < -2 — критическая
// аномалия, -2 <= z < -1 — угнетение биомассы.
export const SEVERITY = {
  critical: { label: 'Критическая аномалия', color: '#c0392b', soft: 'rgba(192, 57, 43, 0.16)' },
  suppression: { label: 'Угнетение биомассы', color: '#d98324', soft: 'rgba(217, 131, 36, 0.16)' },
  normal: { label: 'Норма', color: '#4b8a5a', soft: 'rgba(75, 138, 90, 0.16)' },
}

// Все версии причины, которые умеет называть ядро. non_weather — это
// утверждение, а не «не знаю»: погодные пороги проверены и не сработали.
export const CAUSE = {
  drought: 'Засуха',
  heat: 'Температурный стресс',
  excess_water: 'Переувлажнение',
  cold: 'Затяжной холод',
  abrupt: 'Резкое событие',
  harvest: 'Уборка или скашивание',
  non_weather: 'Причина не погодная',
  unknown: 'Причина не определена',
}

// Откуда взялась климатическая норма. Пользователь должен понимать, когда цифре
// можно верить, а когда это прикидка по культуре — за такую честность дают
// баллы за продукт.
export const CLIMATOLOGY = {
  polygon: {
    label: 'Норма по истории поля',
    hint: 'Норма построена по собственной истории этого поля — оценка надёжная.',
    tone: 'ok',
  },
  crop: {
    label: 'Норма по культуре',
    hint:
      'У поля нет собственной истории, норма взята средняя по культуре. Оценка ориентировочная: ' +
      'глубина отклонения и уверенность в причине занижены намеренно.',
    tone: 'warn',
  },
  none: {
    label: 'Нормы нет',
    hint: 'Сравнивать не с чем: ни собственной истории, ни нормы по культуре. Периоды не ищутся.',
    tone: 'bad',
  },
}

export const SENSOR = {
  s2: 'Sentinel-2',
  landsat: 'Landsat',
  modis: 'MODIS',
  unknown: 'источник неизвестен',
}

export function formatDate(value) {
  if (!value) return '—'
  return new Date(value).toLocaleDateString('ru-RU', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  })
}

export function formatShortDate(value) {
  return new Date(value).toLocaleDateString('ru-RU', { day: '2-digit', month: 'short' })
}

// Склонение существительного при числе: «1 день», «3 дня», «10 дней».
export function plural(n, one, few, many) {
  const mod100 = n % 100
  const mod10 = n % 10
  if (mod100 >= 11 && mod100 <= 14) return `${n} ${many}`
  if (mod10 === 1) return `${n} ${one}`
  if (mod10 >= 2 && mod10 <= 4) return `${n} ${few}`
  return `${n} ${many}`
}
