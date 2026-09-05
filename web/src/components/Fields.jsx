import { useState } from 'react'

import { CLIMATOLOGY, SEVERITY, formatDate } from '../dict.js'
import { IconPencil, IconTrash } from './icons.jsx'

/**
 * Раздел «Участки»: всё управление сохранёнными полями в одном месте.
 *
 * Добавление живёт на карте — участок нельзя завести, не показав, где он.
 * Здесь остальное: посмотреть, переименовать, пересчитать, удалить, открыть на
 * карте. Список отсортирован сервером по глубине худшего отклонения: агроному
 * нужно видеть проблемные поля сверху, а не в порядке добавления.
 */
export default function Fields({ summary, selectedId, onOpen, onRename, onDelete, onAnalyze, onGoMap }) {
  const [editing, setEditing] = useState(null)
  const [name, setName] = useState('')
  const fields = summary?.fields || []

  if (!fields.length) {
    return (
      <div className="card">
        <div className="empty">
          Сохранённых участков пока нет.
          <div style={{ marginTop: 14 }}>
            <button className="btn primary" onClick={onGoMap}>
              Выбрать поле на карте
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="card" style={{ padding: '20px 8px' }}>
      <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Участок</th>
            <th>Площадь</th>
            <th>Культура</th>
            <th>Периодов</th>
            <th>Худшее отклонение</th>
            <th>Норма</th>
            <th>Разобран</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {fields.map((field) => {
            const digest = field.summary
            const worst = digest?.worst_zscore
            const tone = worst == null ? null : worst <= -2 ? SEVERITY.critical : SEVERITY.suppression
            const clim = CLIMATOLOGY[digest?.climatology_source]
            return (
              <tr
                key={field.id}
                className={`pick${field.id === selectedId ? ' sel' : ''}`}
                onClick={() => onOpen(field)}
              >
                <td>
                  {editing === field.id ? (
                    <form
                      onClick={(event) => event.stopPropagation()}
                      onSubmit={(event) => {
                        event.preventDefault()
                        onRename(field.id, name)
                        setEditing(null)
                      }}
                    >
                      <input
                        type="text"
                        className="field"
                        autoFocus
                        value={name}
                        onChange={(event) => setName(event.target.value)}
                        onBlur={() => setEditing(null)}
                      />
                    </form>
                  ) : (
                    <>
                      <div style={{ fontWeight: 600 }}>{field.name}</div>
                      <div className="small muted">
                        {field.source === 'osm' ? 'из OpenStreetMap' : 'нарисован вручную'}
                      </div>
                    </>
                  )}
                </td>
                <td className="num">{field.area_ha} га</td>
                <td className="small">{field.crop_type || <span className="muted">не указана</span>}</td>
                <td className="num">{digest ? digest.anomalies : '—'}</td>
                <td>
                  {tone ? (
                    <span className="tag" style={{ background: tone.soft, color: tone.color }}>
                      {worst.toFixed(1)} σ
                    </span>
                  ) : (
                    <span className="muted small">—</span>
                  )}
                </td>
                <td className="small">
                  {clim ? (
                    <span className="row" style={{ gap: 6 }} title={clim.hint}>
                      <span className={`dot ${clim.tone}`} />
                      {clim.label}
                    </span>
                  ) : (
                    <span className="muted">—</span>
                  )}
                </td>
                <td className="small muted">
                  {field.last_analyzed_at ? formatDate(field.last_analyzed_at) : 'не считался'}
                </td>
                <td onClick={(event) => event.stopPropagation()}>
                  <div className="row" style={{ gap: 2, justifyContent: 'flex-end' }}>
                    <button className="btn ghost sm" onClick={() => onAnalyze(field)}>
                      {digest ? 'Пересчитать' : 'Посчитать'}
                    </button>
                    <button
                      className="btn ghost sm"
                      title="Переименовать"
                      onClick={() => {
                        setEditing(field.id)
                        setName(field.name)
                      }}
                    >
                      <IconPencil />
                    </button>
                    <button
                      className="btn ghost sm danger"
                      title="Удалить"
                      onClick={() => onDelete(field)}
                    >
                      <IconTrash />
                    </button>
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      </div>
    </div>
  )
}
