import { useMemo } from 'react'
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { formatDate } from '../dict.js'
import { usePalette } from '../theme.js'

// Суточные значения за пять сезонов — это полторы тысячи столбиков шириной в
// пиксель, из которых ничего не прочитать. Сворачиваем в декады: осадки
// суммируются, температура усредняется. Такой шаг ещё показывает засушливый
// период, но уже читается глазами.
const BUCKET_DAYS = 10

/**
 * Погода под графиком индекса.
 *
 * Без неё объяснение причины выглядит голословным: фраза «дефицит осадков»
 * должна проверяться взглядом на тот же интервал времени.
 */
export default function WeatherPanel({ weather, anomalies, activeAnomaly }) {
  const palette = usePalette()
  const data = useMemo(() => aggregate(weather), [weather])
  if (!data.length) return null

  return (
    <div className="card">
      <h3>Погода по центроиду поля</h3>
      <div className="legend">
        <span><i className="band" style={{ background: palette.rain }} /> осадки за декаду, мм</span>
        <span><i style={{ borderTopColor: palette.temp }} /> средняя температура, °C</span>
      </div>
      <ResponsiveContainer width="100%" height={170}>
        <ComposedChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: -18 }}>
          <CartesianGrid stroke={palette.grid} vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(t) => new Date(t).toLocaleDateString('ru-RU', { month: 'short', year: '2-digit' })}
            tick={{ fontSize: 11, fill: palette.tick }}
            stroke={palette.axis}
            minTickGap={40}
          />
          {/* Ширина оси считается под три знака: за декаду в ливень набегает
              больше сотни миллиметров, и на 38 пикселях у «140» отрезало
              первую цифру — подпись превращалась в «40». */}
          <YAxis yAxisId="precip" tick={{ fontSize: 11, fill: palette.tick }} stroke={palette.axis} width={46} />
          <YAxis
            yAxisId="temp"
            orientation="right"
            tick={{ fontSize: 11, fill: palette.tick }}
            stroke={palette.axis}
            width={40}
          />

          {anomalies.map((a, index) => {
            const color = a.severity === 'critical' ? palette.critical : palette.suppression
            return (
              <ReferenceArea
                key={`${a.start}-${a.end}`}
                yAxisId="precip"
                x1={Date.parse(a.start)}
                x2={Date.parse(a.end)}
                fill={color}
                fillOpacity={activeAnomaly === index ? 0.26 : 0.12}
              />
            )
          })}

          <Bar yAxisId="precip" dataKey="precip" fill={palette.rain} isAnimationActive={false} />
          <Line
            yAxisId="temp"
            dataKey="temp"
            stroke={palette.temp}
            strokeWidth={1.4}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          <Tooltip content={<WeatherTooltip />} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

function aggregate(weather) {
  if (!weather?.length) return []
  const buckets = new Map()
  const bucketMs = BUCKET_DAYS * 86400000
  for (const point of weather) {
    const t = Date.parse(point.date)
    if (Number.isNaN(t)) continue
    const key = Math.floor(t / bucketMs) * bucketMs
    const bucket = buckets.get(key) || { t: key, precip: 0, tempSum: 0, tempCount: 0 }
    if (point.precip_mm != null) bucket.precip += point.precip_mm
    if (point.temp_c != null) {
      bucket.tempSum += point.temp_c
      bucket.tempCount += 1
    }
    buckets.set(key, bucket)
  }
  return [...buckets.values()]
    .sort((a, b) => a.t - b.t)
    .map((b) => ({
      t: b.t,
      precip: Math.round(b.precip * 10) / 10,
      temp: b.tempCount ? Math.round((b.tempSum / b.tempCount) * 10) / 10 : null,
    }))
}

function WeatherTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  return (
    <div className="card" style={{ margin: 0, padding: '9px 12px' }}>
      <div style={{ fontWeight: 600 }}>декада с {formatDate(point.t)}</div>
      <div className="small">осадки {point.precip} мм</div>
      {point.temp != null && <div className="small">средняя температура {point.temp} °C</div>}
    </div>
  )
}
