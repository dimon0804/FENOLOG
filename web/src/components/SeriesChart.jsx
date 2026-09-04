import { useMemo } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { CAUSE, SENSOR, SEVERITY, formatDate } from '../dict.js'

/**
 * Главный график: ряд NDVI.
 *
 * Пять вещей из постановки, ниже которых демонстрация не засчитывается, тремя
 * из них закрыты здесь: исходный ряд наблюдений, восстановленный ряд и
 * негативные аномальные периоды. Причём исходные и восстановленные значения
 * обязаны различаться **разными визуальными средствами**, а не только цветом:
 * наблюдения — точки, восстановление — сплошная линия.
 */
export default function SeriesChart({ series, anomalies, activeAnomaly, onPickAnomaly }) {
  const data = useMemo(
    () =>
      series.map((point) => ({
        t: Date.parse(point.date),
        observed: point.observed,
        restored: point.restored,
        zscore: point.zscore,
        source: point.source,
        isRestored: point.is_restored,
        // Коридор нормы в два стандартных отклонения: нижняя его граница и есть
        // порог класса «угнетение биомассы», поэтому полоса на графике —
        // не украшение, а объяснение, откуда берутся периоды.
        band:
          point.climatology_mean != null && point.climatology_std != null
            ? [
                point.climatology_mean - 2 * point.climatology_std,
                point.climatology_mean + 2 * point.climatology_std,
              ]
            : null,
      })),
    [series],
  )

  const observedPoints = useMemo(() => data.filter((d) => d.observed != null), [data])
  const hasBand = useMemo(() => data.some((d) => d.band), [data])

  if (!series.length) return null

  return (
    <div className="card">
      <h3>Вегетационный индекс NDVI</h3>
      <div className="legend">
        <span><i className="dotmark" style={{ background: '#23241f' }} /> наблюдения со снимков</span>
        <span><i style={{ borderTopColor: '#3f7d4e' }} /> восстановленный ряд</span>
        {hasBand && <span><i className="band" style={{ background: 'rgba(63,125,78,0.16)' }} /> норма ±2σ</span>}
        <span><i className="band" style={{ background: SEVERITY.suppression.soft }} /> угнетение</span>
        <span><i className="band" style={{ background: SEVERITY.critical.soft }} /> критическая аномалия</span>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
          <CartesianGrid stroke="#e8e5df" vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(t) => new Date(t).toLocaleDateString('ru-RU', { month: 'short', year: '2-digit' })}
            tick={{ fontSize: 11, fill: '#6d6c64' }}
            stroke="#cfcabf"
            minTickGap={40}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: '#6d6c64' }}
            stroke="#cfcabf"
            tickCount={6}
          />

          {/* Периоды рисуются под рядом, чтобы не закрывать его. */}
          {anomalies.map((a, index) => {
            const tone = SEVERITY[a.severity] || SEVERITY.suppression
            const active = activeAnomaly === index
            return (
              <ReferenceArea
                key={`${a.start}-${a.end}`}
                x1={Date.parse(a.start)}
                x2={Date.parse(a.end)}
                fill={tone.color}
                fillOpacity={active ? 0.3 : 0.14}
                stroke={active ? tone.color : 'none'}
                strokeOpacity={0.8}
                onClick={() => onPickAnomaly?.(index)}
                style={{ cursor: 'pointer' }}
              />
            )
          })}

          {hasBand && (
            <Area
              dataKey="band"
              stroke="none"
              fill="#3f7d4e"
              fillOpacity={0.13}
              isAnimationActive={false}
              connectNulls={false}
            />
          )}

          <Line
            dataKey="restored"
            stroke="#3f7d4e"
            strokeWidth={1.8}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />

          {/* Наблюдения отдельным слоем точек: именно это отличает исходные
              значения от восстановленных на глаз, без чтения подписи. */}
          <Scatter
            data={observedPoints}
            dataKey="observed"
            fill="#23241f"
            shape="circle"
            isAnimationActive={false}
          />

          <Tooltip content={<SeriesTooltip anomalies={anomalies} />} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

function SeriesTooltip({ active, payload, anomalies }) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  const inside = anomalies.find(
    (a) => point.t >= Date.parse(a.start) && point.t <= Date.parse(a.end),
  )
  return (
    <div className="card" style={{ margin: 0, padding: '9px 12px', boxShadow: '0 4px 16px rgba(0,0,0,.1)' }}>
      <div style={{ fontWeight: 600 }}>{formatDate(point.t)}</div>
      <div className="small">
        {point.observed != null ? (
          <>наблюдение {point.observed.toFixed(3)} · {SENSOR[point.source] || point.source}</>
        ) : (
          <span className="muted">наблюдения нет, значение восстановлено</span>
        )}
      </div>
      <div className="small">восстановлено {point.restored?.toFixed(3)}</div>
      {point.zscore != null && (
        <div className="small">отклонение от нормы {point.zscore.toFixed(2)} σ</div>
      )}
      {inside && (
        <div className="small" style={{ marginTop: 4, color: (SEVERITY[inside.severity] || {}).color }}>
          {(SEVERITY[inside.severity] || {}).label} · {CAUSE[inside.cause] || inside.cause}
        </div>
      )}
    </div>
  )
}
