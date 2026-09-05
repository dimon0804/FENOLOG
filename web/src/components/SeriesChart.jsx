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

// Физический диапазон NDVI для растительности.
const NDVI_MIN = 0
const NDVI_MAX = 1
const clamp = (v) => Math.min(NDVI_MAX, Math.max(NDVI_MIN, v))

// Вегетационный сезон, месяцы включительно. Те же границы, в которых ядро
// ищет периоды: вне сезона низкий индекс — это снег и голая почва, а не
// состояние посева.
const SEASON = [4, 10]

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
      series.map((point) => {
        const date = new Date(point.date)
        const inSeason = date.getMonth() + 1 >= SEASON[0] && date.getMonth() + 1 <= SEASON[1]
        return {
        t: date.getTime(),
        observed: point.observed,
        restored: point.restored,
        zscore: point.zscore,
        source: point.source,
        isRestored: point.is_restored,
        // Коридор нормы в два стандартных отклонения: нижняя его граница и есть
        // порог класса «угнетение биомассы», поэтому полоса на графике —
        // не украшение, а объяснение, откуда берутся периоды.
        //
        // Рисуется только в сезон, и по двум причинам сразу. Зимой разброс нормы
        // огромен — верхняя граница уходит за 1,4, чего у вегетационного индекса
        // не бывает, — и такая полоса закрывает собой полграфика, не сообщая
        // ничего. Плюс ядро вне сезона периоды и не ищет, так что показывать там
        // норму значило бы обещать сравнение, которого не происходит.
        band:
          inSeason && point.climatology_mean != null && point.climatology_std != null
            ? [
                clamp(point.climatology_mean - 2 * point.climatology_std),
                clamp(point.climatology_mean + 2 * point.climatology_std),
              ]
            : null,
        }
      }),
    [series],
  )

  const observedPoints = useMemo(() => data.filter((d) => d.observed != null), [data])
  const hasBand = useMemo(() => data.some((d) => d.band), [data])

  if (!series.length) return null

  return (
    <div className="card">
      <h3>Вегетационный индекс NDVI</h3>
      <div className="legend">
        <span><i className="dotmark" style={{ background: '#101010' }} /> наблюдения со снимков</span>
        <span><i style={{ borderTopColor: '#2f6b2a' }} /> восстановленный ряд</span>
        {hasBand && <span><i className="band" style={{ background: 'rgba(47,107,42,0.16)' }} /> норма ±2σ, апрель—октябрь</span>}
        <span><i className="band" style={{ background: SEVERITY.suppression.soft }} /> угнетение</span>
        <span><i className="band" style={{ background: SEVERITY.critical.soft }} /> критическая аномалия</span>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="#eeeeee" vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(t) => new Date(t).toLocaleDateString('ru-RU', { month: 'short', year: '2-digit' })}
            tick={{ fontSize: 11, fill: '#6f6f6f' }}
            stroke="#dddddd"
            minTickGap={40}
          />
          <YAxis
            domain={[NDVI_MIN, NDVI_MAX]}
            // Без allowDataOverflow полоса нормы растягивает шкалу за свои
            // пределы, и на оси появляются подписи вроде 1,49 и −0,000006.
            allowDataOverflow
            tickFormatter={(v) => v.toFixed(1)}
            tick={{ fontSize: 11, fill: '#6f6f6f' }}
            stroke="#dddddd"
            tickCount={6}
            width={34}
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
              fill="#2f6b2a"
              fillOpacity={0.13}
              isAnimationActive={false}
              connectNulls={false}
            />
          )}

          <Line
            dataKey="restored"
            stroke="#2f6b2a"
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
            fill="#101010"
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
