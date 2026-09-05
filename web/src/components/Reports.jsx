import { useState } from 'react'

import { api } from '../api.js'
import { CAUSE, SEVERITY, formatDate, plural } from '../dict.js'
import { IconDownload } from './icons.jsx'

/**
 * Раздел «Отчёты»: выгрузка разбора поля файлом.
 *
 * То, что агроном реально унесёт с собой и покажет агрономической службе:
 * таблица ряда и таблица найденных периодов с готовыми объяснениями. Формат
 * CSV, потому что открывается в Excel без плясок, плюс JSON для тех, кто будет
 * считать дальше.
 *
 * Файл собирается в браузере из уже полученного разбора: гонять его на сервер
 * ради переформатирования незачем, а лишний эндпоинт — лишний способ сломаться.
 */
export default function Reports({ summary, onGoMap }) {
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)
  const fields = (summary?.fields || []).filter((f) => f.summary)

  if (!fields.length) {
    return (
      <div className="card">
        <div className="empty">
          Выгружать пока нечего: ни одно поле не разобрано.
          <div style={{ marginTop: 14 }}>
            <button className="btn primary" onClick={onGoMap}>
              Выбрать поле на карте
            </button>
          </div>
        </div>
      </div>
    )
  }

  async function download(field, kind) {
    setBusy(`${field.id}:${kind}`)
    setError(null)
    try {
      const payload = await api.savedResult(field.id)
      const stamp = new Date().toISOString().slice(0, 10)
      const safe = field.name.replace(/[^\wа-яА-ЯёЁ -]+/g, '').trim().replace(/\s+/g, '_')
      if (kind === 'json') {
        save(`${safe}_${stamp}.json`, JSON.stringify(payload, null, 2), 'application/json')
      } else if (kind === 'series') {
        save(`${safe}_ряд_${stamp}.csv`, seriesCsv(payload.result.series), 'text/csv')
      } else {
        save(`${safe}_периоды_${stamp}.csv`, anomaliesCsv(payload.result.anomalies), 'text/csv')
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      {error && <div className="banner" style={{ marginBottom: 16 }}>{error}</div>}

      {fields.map((field) => {
        const digest = field.summary
        return (
          <div className="card" key={field.id}>
            <div className="row wrap" style={{ justifyContent: 'space-between' }}>
              <div>
                <h3 style={{ marginBottom: 2 }}>{field.name}</h3>
                <div className="small muted">
                  {field.area_ha} га{field.crop_type ? ` · ${field.crop_type}` : ''} ·{' '}
                  {formatDate(digest.date_from)} — {formatDate(digest.date_to)} ·{' '}
                  {plural(digest.observations, 'наблюдение', 'наблюдения', 'наблюдений')} ·{' '}
                  разбор от {formatDate(field.last_analyzed_at)}
                </div>
              </div>
              <div className="row" style={{ gap: 8 }}>
                <button
                  className="btn"
                  disabled={busy === `${field.id}:series`}
                  onClick={() => download(field, 'series')}
                >
                  <IconDownload width={17} height={17} />
                  Ряд, CSV
                </button>
                <button
                  className="btn"
                  disabled={busy === `${field.id}:anomalies`}
                  onClick={() => download(field, 'anomalies')}
                >
                  <IconDownload width={17} height={17} />
                  Периоды, CSV
                </button>
                <button
                  className="btn"
                  disabled={busy === `${field.id}:json`}
                  onClick={() => download(field, 'json')}
                >
                  <IconDownload width={17} height={17} />
                  Всё, JSON
                </button>
              </div>
            </div>

            <div className="row wrap" style={{ gap: 10, marginTop: 14 }}>
              <span className="chip">
                {plural(digest.anomalies, 'период', 'периода', 'периодов')}
              </span>
              {digest.critical > 0 && (
                <span className="chip" style={{ color: SEVERITY.critical.color }}>
                  критических: {digest.critical}
                </span>
              )}
              {digest.suppression > 0 && (
                <span className="chip" style={{ color: SEVERITY.suppression.color }}>
                  угнетения: {digest.suppression}
                </span>
              )}
              {digest.worst_zscore != null && (
                <span className="chip">худшее отклонение {digest.worst_zscore.toFixed(1)} σ</span>
              )}
              {digest.failures?.length > 0 && (
                <span className="chip warn">часть данных не собралась</span>
              )}
            </div>
          </div>
        )
      })}
    </>
  )
}

// Точка с запятой и BOM — чтобы Excel с русской локалью открыл файл сразу, а не
// свалил всё в один столбец и не показал кракозябры вместо кириллицы.
const SEP = ';'

function csv(rows) {
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

function seriesCsv(series) {
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

function anomaliesCsv(anomalies) {
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

// Десятичная запятая: с точкой Excel в русской локали считает число текстом.
function num(value) {
  return value == null ? '' : String(value).replace('.', ',')
}

function save(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type: `${type};charset=utf-8` }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}
