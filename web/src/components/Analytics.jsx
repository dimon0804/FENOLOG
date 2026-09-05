import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { api } from '../api.js'
import { SEVERITY, formatDate, plural } from '../dict.js'

// Цвета линий сравнения. Намеренно не из смысловой тройки и не из палитры
// классов аномалий: здесь цвет означает всего лишь «поле номер такой-то», и
// путать его с «критическая аномалия» нельзя.
const LINES = ['#2f6b2a', '#1f34d4', '#8b2fe8', '#c46a12', '#0f8f8f']
const MAX_COMPARE = 5

/**
 * Раздел «Аналитика»: сводка по всем разобранным полям и сравнение их динамики
 * на одном графике.
 *
 * Это то, ради чего сервис перестаёт быть инструментом по одному полю: видно,
 * что творится в хозяйстве целиком и какое поле выбивается из общей картины.
 */
export default function Analytics({ summary, onOpenField }) {
  const analyzed = useMemo(
    () => (summary?.fields || []).filter((f) => f.summary),
    [summary],
  )
  const [picked, setPicked] = useState([])
  const [series, setSeries] = useState({})
  const [loading, setLoading] = useState(false)

  // По умолчанию берём три худших поля: именно их и хочется сравнить первыми.
  useEffect(() => {
    setPicked((current) =>
      current.length ? current : analyzed.slice(0, 3).map((f) => f.id),
    )
  }, [analyzed])

  useEffect(() => {
    const missing = picked.filter((id) => !(id in series))
    if (!missing.length) return
    setLoading(true)
    Promise.all(
      missing.map((id) =>
        api
          .savedResult(id)
          .then((payload) => [id, payload.result.series])
          .catch(() => [id, null]),
      ),
    )
      .then((pairs) => setSeries((prev) => ({ ...prev, ...Object.fromEntries(pairs) })))
      .finally(() => setLoading(false))
  }, [picked, series])

  const chart = useMemo(() => buildChart(picked, series, analyzed), [picked, series, analyzed])

  if (!analyzed.length) {
    return (
      <div className="card">
        <div className="empty">
          Сравнивать пока нечего: ни одно поле не разобрано. Выберите участок на карте и
          запустите анализ.
        </div>
      </div>
    )
  }

  const totals = summary.anomalies

  return (
    <>
      <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 16 }}>
        <Mini label="Полей разобрано" value={`${summary.analyzed} из ${summary.polygons}`} />
        <Mini label="Периодов найдено" value={totals.total} />
        <Mini label="Критических" value={totals.critical} tone="var(--critical)" />
        <Mini label="Угнетения биомассы" value={totals.suppression} tone="var(--suppression)" />
      </div>

      <div className="card" style={{ marginTop: 18 }}>
        <h3>Сравнение полей</h3>
        <p className="small muted" style={{ marginTop: -8 }}>
          Восстановленные ряды выбранных полей на одной шкале. Больше {MAX_COMPARE} линий на
          графике уже не читаются, поэтому выбор ограничен.
        </p>

        <div className="row wrap" style={{ gap: 8, marginBottom: 14 }}>
          {analyzed.map((field, index) => {
            const on = picked.includes(field.id)
            const color = LINES[picked.indexOf(field.id) % LINES.length]
            return (
              <button
                key={field.id}
                className="chip"
                style={{
                  cursor: 'pointer',
                  borderColor: on ? color : 'var(--line)',
                  color: on ? color : 'var(--ink)',
                  fontWeight: on ? 600 : 400,
                }}
                onClick={() =>
                  setPicked((current) =>
                    current.includes(field.id)
                      ? current.filter((id) => id !== field.id)
                      : current.length >= MAX_COMPARE
                        ? current
                        : [...current, field.id],
                  )
                }
              >
                {on && <span className="dot" style={{ background: color }} />}
                {field.name}
              </button>
            )
          })}
        </div>

        {loading && <div className="small muted">Загружаю ряды…</div>}

        {chart.rows.length > 0 && (
          <ResponsiveContainer width="100%" height={330}>
            <LineChart data={chart.rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="#eee" vertical={false} />
              <XAxis
                dataKey="t"
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(t) =>
                  new Date(t).toLocaleDateString('ru-RU', { month: 'short', year: '2-digit' })}
                tick={{ fontSize: 11, fill: '#6f6f6f' }}
                stroke="#ddd"
                minTickGap={40}
              />
              <YAxis
                domain={[0, 1]}
                allowDataOverflow
                tickFormatter={(v) => v.toFixed(1)}
                tick={{ fontSize: 11, fill: '#6f6f6f' }}
                stroke="#ddd"
                width={34}
              />
              <Tooltip
                labelFormatter={(t) => formatDate(t)}
                formatter={(value, name) => [value?.toFixed(3), name]}
                contentStyle={{ borderRadius: 10, border: '1px solid #eee', fontSize: 13 }}
              />
              <Legend wrapperStyle={{ fontSize: 12.5 }} />
              {chart.keys.map((key, index) => (
                <Line
                  key={key.id}
                  dataKey={key.id}
                  name={key.name}
                  stroke={LINES[index % LINES.length]}
                  strokeWidth={1.8}
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="card">
        <h3>Поля по глубине отклонения</h3>
        {analyzed.map((field) => {
          const worst = field.summary.worst_zscore
          const tone = worst == null ? null : worst <= -2 ? SEVERITY.critical : SEVERITY.suppression
          return (
            <div
              key={field.id}
              className="row"
              style={{ padding: '11px 0', borderTop: '1px solid var(--line)', cursor: 'pointer' }}
              onClick={() => onOpenField(field)}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontWeight: 600 }}>{field.name}</div>
                <div className="small muted">
                  {field.area_ha} га · {plural(field.summary.observations, 'наблюдение', 'наблюдения', 'наблюдений')}
                  {' · '}
                  {formatDate(field.summary.date_from)} — {formatDate(field.summary.date_to)}
                </div>
              </div>
              <span className="small muted">
                {plural(field.summary.anomalies, 'период', 'периода', 'периодов')}
              </span>
              {tone && (
                <span className="tag" style={{ background: tone.soft, color: tone.color }}>
                  {worst.toFixed(1)} σ
                </span>
              )}
            </div>
          )
        })}
      </div>
    </>
  )
}

function Mini({ label, value, tone }) {
  return (
    <div className="card" style={{ margin: 0, padding: '18px 20px' }}>
      <div className="small muted">{label}</div>
      <div style={{ fontSize: 30, fontWeight: 800, marginTop: 4, color: tone || 'var(--ink)' }}>
        {value}
      </div>
    </div>
  )
}

/**
 * Сборка данных для общего графика.
 *
 * Ряды разных полей посчитаны на своих сетках дат, и просто склеить их нельзя.
 * Раскладываем по общей шкале времени: одна строка на дату, в ней значения тех
 * полей, у которых на эту дату что-то есть. Recharts с connectNulls дорисует
 * разрывы сам.
 */
function buildChart(picked, series, fields) {
  const keys = picked
    .map((id) => fields.find((f) => f.id === id))
    .filter(Boolean)
    .map((f) => ({ id: f.id, name: f.name }))

  const byDate = new Map()
  for (const id of picked) {
    for (const point of series[id] || []) {
      const t = Date.parse(point.date)
      const row = byDate.get(t) || { t }
      row[id] = point.restored
      byDate.set(t, row)
    }
  }
  return { keys, rows: [...byDate.values()].sort((a, b) => a.t - b.t) }
}
