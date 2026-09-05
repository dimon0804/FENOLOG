/**
 * Выгрузка таблиц файлом.
 *
 * Собирается в браузере из уже полученного разбора: гонять его на сервер ради
 * переформатирования незачем, а лишний эндпоинт — лишний способ сломаться.
 *
 * Формат подогнан под Excel с русской локалью: разделитель — точка с запятой,
 * десятичный знак — запятая, кодировка UTF-8 с BOM. С запятой-разделителем
 * Excel сваливает всё в один столбец, без BOM — показывает кракозябры вместо
 * кириллицы, с десятичной точкой — считает числа текстом.
 */
import { CAUSE, SEVERITY } from './dict.js'

const SEP = ';'

export function csv(rows) {
  return (
    '﻿' +
    rows
      .map((row) =>
        row
          .map((cell) => {
            const text = cell == null ? '' : String(cell)
            return /[";\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
          })
          .join(SEP),
      )
      .join('\r\n')
  )
}

export function seriesCsv(series) {
  return csv([
    ['дата', 'наблюдение', 'восстановлено', 'норма', 'станд_откл', 'z_оценка', 'восстановлено_да_нет', 'сенсор'],
    ...series.map((p) => [
      p.date,
      num(p.observed),
      num(p.restored),
      num(p.climatology_mean),
      num(p.climatology_std),
      num(p.zscore),
      p.is_restored ? 'восстановлено' : 'наблюдение',
      p.source || '',
    ]),
  ])
}

export function anomaliesCsv(anomalies) {
  return csv([
    ['начало', 'конец', 'дней', 'класс', 'причина', 'уверенность', 'z_минимум', 'z_среднее', 'объяснение'],
    ...anomalies.map((a) => [
      a.start,
      a.end,
      a.duration_days,
      SEVERITY[a.severity]?.label || a.severity,
      CAUSE[a.cause] || a.cause,
      num(a.cause_confidence),
      num(a.min_zscore),
      num(a.mean_zscore),
      a.explanation,
    ]),
  ])
}

/** Десятичная запятая: с точкой Excel в русской локали считает число текстом. */
export function num(value) {
  return value == null ? '' : String(value).replace('.', ',')
}

/** Имя файла без символов, на которых спотыкается файловая система. */
export function safeName(name) {
  return name.replace(/[^\wа-яА-ЯёЁ -]+/g, '').trim().replace(/\s+/g, '_')
}

export function save(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type: `${type};charset=utf-8` }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}
