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
import { safeName, save, seriesCsv } from '../csv.js'
import { FIELD_STATE, formatDate } from '../dict.js'
import { IconDownload } from './icons.jsx'

// Цвета линий сравнения. Намеренно не из смысловой тройки и не из палитры
// классов аномалий: здесь цвет означает всего лишь «поле номер такой-то», и
// путать его с «критическая аномалия» нельзя.
const LINES = ['#2f6b2a', '#1f34d4', '#8b2fe8', '#c46a12', '#0f8f8f']
const MAX_COMPARE = 5

// Зоны состояния — те же пороги, по которым ядро выделяет периоды: глубже двух
// сигм аномалия, от одной до двух повод присмотреться. Названия и цвета берутся
// из общего словаря состояний, того же, которым карта красит контуры: одно и то
// же состояние поля обязано называться на всех экранах одинаково.
const ZONES = [
  { key: 'ok', test: (z) => z > -1 },
  { key: 'watch', test: (z) => z > -2 },
  { key: 'bad', test: () => true },
].map((zone) => ({ ...zone, ...FIELD_STATE[zone.key] }))

/**
 * Раздел «Аналитика»: разбор одного поля по сезонам плюс сравнение полей.
 *
 * Экран отвечает на два разных вопроса, и потому разделён надвое. Сверху —
 * выбранное поле в выбранном сезоне: средний индекс, худшее отклонение, сколько
 * дней поле провело в аномалии, сколько выпало осадков, и как всё это
 * распределено по зонам состояния. Снизу — сравнение полей между собой: то,
 * ради чего сервис перестаёт быть инструментом по одному полю.
 */
export default function Analytics({ summary, onOpenField }) {
  const analyzed = useMemo(
    () => (summary?.fields || []).filter((f) => f.summary),
    [summary],
  )

  const [fieldId, setFieldId] = useState(null)
  // null — сезон ещё не выбран: подставим последний, как только придёт разбор.
  const [season, setSeason] = useState(null)
  const [series, setSeries] = useState({})
  const [payloads, setPayloads] = useState({})
  const [loading, setLoading] = useState(false)

  // Поле по умолчанию — первое в списке, а список сервер сортирует по глубине
  // худшего отклонения: открывать аналитику на самом спокойном поле незачем.
  useEffect(() => {
    if (!fieldId && analyzed.length) setFieldId(analyzed[0].id)
  }, [analyzed, fieldId])

  useEffect(() => {
    if (!fieldId || fieldId in payloads) return
    api
      .savedResult(fieldId)
      .then((payload) => setPayloads((prev) => ({ ...prev, [fieldId]: payload })))
      .catch(() => setPayloads((prev) => ({ ...prev, [fieldId]: null })))
  }, [fieldId, payloads])

  const field = analyzed.find((f) => f.id === fieldId) || null
  const payload = fieldId ? payloads[fieldId] : null
  const seasons = useMemo(() => seasonsOf(payload), [payload])
  const window = useMemo(() => windowOf(seasons, season), [seasons, season])
  const stats = useMemo(() => summarize(payload, window), [payload, window])

  // По умолчанию — последний сезон, а не весь ряд: за пять лет «осадки 3387 мм»
  // и «247 дней в аномалии» верны, но ни о чём не говорят. Показатели имеют
  // смысл в пределах сезона, с ним же есть с чем сравнивать.
  //
  // Выбранный сезон может не существовать у другого поля — тогда откатываемся.
  useEffect(() => {
    if (!seasons.length) return
    if (season === null || (season !== 'all' && !seasons.includes(Number(season)))) {
      setSeason(seasons[0])
    }
  }, [seasons, season])

  if (!analyzed.length) {
    return (
      <div className="card">
        <div className="empty">
          Разбирать пока нечего: ни одно поле не проанализировано. Выберите участок на карте
          и запустите анализ.
        </div>
      </div>
    )
  }

  function exportSeries() {
    if (!stats) return
    const stamp = new Date().toISOString().slice(0, 10)
    const label = season === 'all' ? 'весь_ряд' : season
    save(`${safeName(field.name)}_${label}_${stamp}.csv`, seriesCsv(stats.points), 'text/csv')
  }

  return (
    <>
      <div className="fields-bar">
        <select
          className="select"
          value={fieldId || ''}
          onChange={(event) => setFieldId(event.target.value)}
        >
          {analyzed.map((f) => (
            <option key={f.id} value={f.id}>{f.name}</option>
          ))}
        </select>

        <select
          className="select"
          value={season ?? ''}
          onChange={(event) => setSeason(event.target.value)}
          disabled={!seasons.length}
        >
          <option value="all">Весь ряд</option>
          {seasons.map((year) => (
            <option key={year} value={year}>Сезон {year}</option>
          ))}
        </select>

        <button className="btn add" onClick={exportSeries} disabled={!stats}>
          <IconDownload width={17} height={17} />
          Экспорт
        </button>
      </div>

      {!payload && <div className="card"><div className="empty">Загружаю разбор…</div></div>}

      {stats && (
        <>
          <div className="kpi-grid">
            <Kpi
              label="NDVI, среднее"
              value={stats.meanNdvi?.toFixed(2)}
              note={stats.meanDelta}
              spark={<Spark values={stats.spark} color="#2f6b2a" />}
            />
            <Kpi
              label="Худшее отклонение"
              value={stats.worstZ?.toFixed(1)}
              unit="σ"
              note={
                stats.worstAt
                  ? { text: formatDate(stats.worstAt), tone: 'muted' }
                  : { text: 'отклонений нет', tone: 'muted' }
              }
              spark={<Spark values={stats.sparkZ} color="#d4342a" />}
            />
            <Kpi
              label="Дней с отклонением"
              value={stats.anomalyDays}
              note={{
                text: `${Math.round((stats.anomalyDays / Math.max(stats.days, 1)) * 100)}% дней периода`,
                tone: stats.anomalyDays ? 'bad' : 'muted',
              }}
              spark={<Bars values={stats.anomalyByMonth} color="#d4342a" />}
            />
            <Kpi
              label="Осадки"
              value={stats.rain == null ? '—' : Math.round(stats.rain)}
              unit={stats.rain == null ? '' : 'мм'}
              note={stats.rainDelta}
              spark={<Bars values={stats.rainByMonth} color="#4e9b36" />}
            />
          </div>

          <div className="analytics-row">
            <div className="card">
              <h3>Динамика NDVI</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={stats.chart} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke="#eee" vertical={false} />
                  <XAxis
                    dataKey="t"
                    type="number"
                    scale="time"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={(t) =>
                      new Date(t).toLocaleDateString(
                        'ru-RU',
                        // Внутри сезона число и месяц, на длинном ряду — месяц и
                        // год: «13.04» четыре раза подряд ничего не различает.
                        stats.days > 400
                          ? { month: 'short', year: '2-digit' }
                          : { day: '2-digit', month: '2-digit' },
                      )}
                    tick={{ fontSize: 11, fill: '#6f6f6f' }}
                    stroke="#ddd"
                    minTickGap={44}
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
                    formatter={(value) => [value?.toFixed(3), 'NDVI']}
                    contentStyle={{ borderRadius: 10, border: '1px solid #eee', fontSize: 13 }}
                  />
                  <Line
                    dataKey="v"
                    stroke="#2f6b2a"
                    strokeWidth={1.8}
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <h3>Распределение по зонам</h3>
              <p className="small muted" style={{ marginTop: -8 }}>
                Доля дней периода по глубине отклонения от нормы поля.
              </p>
              <div className="zones">
                <Donut parts={stats.zones} />
                <div className="zone-legend">
                  {stats.zones.map((zone) => (
                    <div key={zone.key} className="zone-row">
                      <span className="dot" style={{ background: zone.color }} />
                      <span className="zone-name">{zone.label}</span>
                      <b>{zone.share.toFixed(1)}%</b>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      <Compare analyzed={analyzed} series={series} setSeries={setSeries} loading={loading} setLoading={setLoading} onOpenField={onOpenField} />
    </>
  )
}

/** Карточка показателя: подпись, число, строка изменения и мини-график. */
function Kpi({ label, value, unit, note, spark }) {
  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className="value">
        {value ?? '—'}
        {unit ? <span className="unit"> {unit}</span> : null}
      </div>
      <div className={`note ${note?.tone || 'muted'}`}>{note?.text || ''}</div>
      <div className="spark">{spark}</div>
    </div>
  )
}

/** Мини-график линией. Осей и подписей нет намеренно: важна только форма. */
function Spark({ values, color }) {
  if (!values || values.length < 2) return null
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const points = values
    .map((v, i) => `${(i / (values.length - 1)) * 100},${28 - ((v - min) / span) * 26}`)
    .join(' ')
  return (
    <svg viewBox="0 0 100 30" preserveAspectRatio="none" width="100%" height="46">
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

/** Мини-график столбцами: для величин, которые складываются, а не текут. */
function Bars({ values, color }) {
  if (!values || !values.length) return null
  const max = Math.max(...values, 1)
  const step = 100 / values.length
  return (
    <svg viewBox="0 0 100 30" preserveAspectRatio="none" width="100%" height="46">
      {values.map((v, i) => (
        <rect
          key={i}
          x={i * step + step * 0.18}
          y={30 - Math.max((v / max) * 28, v > 0 ? 2 : 0)}
          width={step * 0.64}
          height={Math.max((v / max) * 28, v > 0 ? 2 : 0)}
          fill={color}
        />
      ))}
    </svg>
  )
}

/** Кольцевая диаграмма долей. Дуги рисуются штрихом по одной окружности. */
function Donut({ parts }) {
  const R = 52
  const C = 2 * Math.PI * R
  let offset = 0
  return (
    <svg width="140" height="140" viewBox="0 0 140 140">
      <circle cx="70" cy="70" r={R} fill="none" stroke="#ececec" strokeWidth="26" />
      {parts.map((part) => {
        const len = (part.share / 100) * C
        const dash = `${len} ${C - len}`
        const node = (
          <circle
            key={part.key}
            cx="70"
            cy="70"
            r={R}
            fill="none"
            stroke={part.color}
            strokeWidth="26"
            strokeDasharray={dash}
            strokeDashoffset={-offset}
            transform="rotate(-90 70 70)"
          />
        )
        offset += len
        return node
      })}
    </svg>
  )
}

/** Сравнение полей: восстановленные ряды нескольких полей на одной шкале. */
function Compare({ analyzed, series, setSeries, loading, setLoading, onOpenField }) {
  const [picked, setPicked] = useState([])

  useEffect(() => {
    setPicked((current) => (current.length ? current : analyzed.slice(0, 3).map((f) => f.id)))
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
  }, [picked, series, setSeries, setLoading])

  const chart = useMemo(() => buildChart(picked, series, analyzed), [picked, series, analyzed])

  if (analyzed.length < 2) return null

  return (
    <div className="card">
      <h3>Сравнение полей</h3>
      <p className="small muted" style={{ marginTop: -8 }}>
        Восстановленные ряды выбранных полей на одной шкале. Больше {MAX_COMPARE} линий на
        графике уже не читаются, поэтому выбор ограничен.
      </p>

      <div className="row wrap" style={{ gap: 8, marginBottom: 14 }}>
        {analyzed.map((field) => {
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
              onDoubleClick={() => onOpenField(field)}
            >
              {on && <span className="dot" style={{ background: color }} />}
              {field.name}
            </button>
          )
        })}
      </div>

      {loading && <div className="small muted">Загружаю ряды…</div>}

      {chart.rows.length > 0 && (
        <ResponsiveContainer width="100%" height={320}>
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
  )
}

// --------------------------------------------------------------------- расчёты

/** Годы, за которые в ряду есть данные вегетационного сезона. */
function seasonsOf(payload) {
  const series = payload?.result?.series || []
  return [...new Set(series.map((p) => Number(p.date.slice(0, 4))))].sort((a, b) => b - a)
}

function windowOf(seasons, season) {
  if (season === 'all' || season === null || !seasons.length) return null
  return { from: `${season}-01-01`, to: `${season}-12-31` }
}

/**
 * Показатели выбранного поля за выбранный период.
 *
 * Всё считается по уже полученному разбору, без обращений к серверу: ряд, нормы
 * и найденные периоды уже лежат в ответе, и пересчитывать их на бэкенде значило
 * бы гонять те же числа туда и обратно.
 */
function summarize(payload, window) {
  const result = payload?.result
  if (!result?.series?.length) return null

  const inWindow = (date) => !window || (date >= window.from && date <= window.to)
  const points = result.series.filter((p) => inWindow(p.date))
  if (!points.length) return null

  const from = points[0].date
  const to = points[points.length - 1].date
  const days = Math.max(1, Math.round((Date.parse(to) - Date.parse(from)) / 86400000) + 1)

  const restored = points.map((p) => p.restored).filter((v) => v != null)
  const meanNdvi = restored.length ? restored.reduce((a, b) => a + b, 0) / restored.length : null

  // Тот же показатель за предыдущий сезон — чтобы «0,72» не висело без опоры.
  // Окно сравнения берётся не «весь прошлый год», а тот же кусок календаря:
  // текущий сезон в сентябре ещё не кончился, и сравнивать его сумму осадков с
  // полным прошлым годом значит каждый раз показывать выдуманный провал.
  const prev = window ? sameWindowLastYear(from, to) : null
  const prevPoints = prev
    ? result.series.filter((p) => p.date >= prev.from && p.date <= prev.to)
    : []
  const prevRestored = prevPoints.map((p) => p.restored).filter((v) => v != null)
  const prevMean = prevRestored.length
    ? prevRestored.reduce((a, b) => a + b, 0) / prevRestored.length
    : null

  const zs = points.map((p) => p.zscore).filter((v) => v != null)
  const worstZ = zs.length ? Math.min(...zs) : null
  const worstAt = worstZ == null ? null : points.find((p) => p.zscore === worstZ)?.date

  const anomalies = (result.anomalies || []).filter((a) => inWindow(a.start) || inWindow(a.end))
  const anomalyDays = anomalies.reduce((sum, a) => sum + (a.duration_days || 0), 0)

  const weather = (payload.result?.weather || payload.weather || []).filter((w) => inWindow(w.date))
  const rainValues = weather.map((w) => w.precip_mm).filter((v) => v != null)
  const rain = rainValues.length ? rainValues.reduce((a, b) => a + b, 0) : null

  const prevWeather = prev
    ? (payload.result?.weather || payload.weather || []).filter(
        (w) => w.date >= prev.from && w.date <= prev.to,
      )
    : []
  const prevRainValues = prevWeather.map((w) => w.precip_mm).filter((v) => v != null)
  const prevRain = prevRainValues.length ? prevRainValues.reduce((a, b) => a + b, 0) : null

  return {
    points,
    from,
    to,
    days,
    meanNdvi,
    meanDelta: delta(meanNdvi, prevMean, 2, '', 'к прошлому сезону'),
    worstZ,
    worstAt,
    anomalyDays,
    anomalyByMonth: byMonth(anomalies, from, to),
    rain,
    rainDelta: delta(rain, prevRain, 0, ' мм', 'к прошлому сезону'),
    rainByMonth: rainMonths(weather, from, to),
    spark: thin(restored, 60),
    sparkZ: thin(zs, 60),
    chart: points.map((p) => ({ t: Date.parse(p.date), v: p.restored })),
    zones: zones(points),
  }
}

function sameWindowLastYear(from, to) {
  const shift = (date) => `${Number(date.slice(0, 4)) - 1}${date.slice(4)}`
  return { from: shift(from), to: shift(to) }
}

/** Строка изменения под числом: со знаком, цветом и понятной привязкой. */
function delta(value, previous, digits, unit, what) {
  if (value == null) return { text: '', tone: 'muted' }
  if (previous == null) return { text: 'сравнить не с чем', tone: 'muted' }
  const diff = value - previous
  const sign = diff > 0 ? '+' : diff < 0 ? '−' : '±'
  return {
    text: `${sign}${Math.abs(diff).toFixed(digits)}${unit} ${what}`,
    tone: diff >= 0 ? 'up' : 'down',
  }
}

/** Доли дней периода по зонам состояния. */
function zones(points) {
  const counts = { ok: 0, watch: 0, bad: 0 }
  let total = 0
  for (const point of points) {
    if (point.zscore == null) continue
    total += 1
    const zone = ZONES.find((z) => z.test(point.zscore))
    counts[zone.key] += 1
  }
  return ZONES.map((zone) => ({
    ...zone,
    share: total ? (counts[zone.key] / total) * 100 : 0,
  }))
}

/** Дни аномалий, разложенные по месяцам периода — для столбиков на карточке. */
function byMonth(anomalies, from, to) {
  const months = monthKeys(from, to)
  const bucket = new Map(months.map((m) => [m, 0]))
  for (const anomaly of anomalies) {
    const key = anomaly.start.slice(0, 7)
    if (bucket.has(key)) bucket.set(key, bucket.get(key) + (anomaly.duration_days || 0))
  }
  return months.map((m) => bucket.get(m))
}

function rainMonths(weather, from, to) {
  const months = monthKeys(from, to)
  const bucket = new Map(months.map((m) => [m, 0]))
  for (const day of weather) {
    const key = day.date.slice(0, 7)
    if (bucket.has(key) && day.precip_mm != null) {
      bucket.set(key, bucket.get(key) + day.precip_mm)
    }
  }
  return months.map((m) => bucket.get(m))
}

function monthKeys(from, to) {
  const keys = []
  const cursor = new Date(from.slice(0, 7) + '-01T00:00:00Z')
  const end = new Date(to.slice(0, 7) + '-01T00:00:00Z')
  while (cursor <= end && keys.length < 200) {
    keys.push(cursor.toISOString().slice(0, 7))
    cursor.setUTCMonth(cursor.getUTCMonth() + 1)
  }
  return keys
}

/** Прореживание ряда для мини-графика: в 46 пикселях высоты 1600 точек лишние. */
function thin(values, limit) {
  if (values.length <= limit) return values
  const step = values.length / limit
  return Array.from({ length: limit }, (_, i) => values[Math.floor(i * step)])
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
