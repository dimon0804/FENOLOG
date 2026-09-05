import { useState } from 'react'

import { api } from '../api.js'
import { anomaliesCsv, safeName, save, seriesCsv } from '../csv.js'
import { formatDate, plural } from '../dict.js'
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
      const safe = safeName(field.name)
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
              {/* Кнопок четыре, и в узком окне они обязаны переноситься:
                  без wrap ряд не влезал в карточку и обрезался по краю. */}
              <div className="row wrap" style={{ gap: 8 }}>
                {/* Отчёт для человека идёт первым и выделен: выгрузки рядом —
                    это данные для аналитика, а PDF читают без подготовки. */}
                <a
                  className="btn primary"
                  href={`/api/polygons/${field.id}/report.pdf`}
                  download
                  title="PDF с графиками и объяснением обычными словами"
                >
                  <IconDownload width={17} height={17} />
                  Отчёт PDF
                </a>
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
                <span className="chip tone-critical">критических: {digest.critical}</span>
              )}
              {digest.suppression > 0 && (
                <span className="chip tone-suppression">угнетения: {digest.suppression}</span>
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
