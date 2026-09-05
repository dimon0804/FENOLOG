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
import { usePalette } from '../theme.js'

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
  // Цвета осей, сетки и самого ряда зависят от темы, а задаются атрибутами
  // SVG — переменная CSS туда не подставляется, поэтому значения читаются из
  // тех же переменных заранее (см. src/theme.js).
  const palette = usePalette()
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
        <span><i className="dotmark" style={{ background: palette.observed }} /> наблюдения со снимков</span>
        <span><i style={{ borderTopColor: palette.series }} /> восстановленный ряд</span>
        {hasBand && <span><i className="band" style={{ background: palette.band, opacity: 0.4 }} /> норма ±2σ, апрель—октябрь</span>}
        <span><i className="band" style={{ background: palette.suppression, opacity: 0.3 }} /> угнетение</span>
        <span><i className="band" style={{ background: palette.critical, opacity: 0.3 }} /> критическая аномалия</span>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
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
          <YAxis
            domain={[NDVI_MIN, NDVI_MAX]}
            // Без allowDataOverflow полоса нормы растягивает шкалу за свои
            // пределы, и на оси появляются подписи вроде 1,49 и −0,000006.
            allowDataOverflow
            tickFormatter={(v) => v.toFixed(1)}
            tick={{ fontSize: 11, fill: palette.tick }}
            stroke={palette.axis}
            tickCount={6}
            width={34}
          />

          {/* Периоды рисуются под рядом, чтобы не закрывать его. */}
          {anomalies.map((a, index) => {
            // Цвет периода берётся из палитры темы по классу аномалии:
            // в тёмной теме те же классы окрашены светлее, иначе заливка
            // на тёмном графике не видна вовсе.
            const color = a.severity === 'critical' ? palette.critical : palette.suppression
            const active = activeAnomaly === index
            return (
              <ReferenceArea
                key={`${a.start}-${a.end}`}
                x1={Date.parse(a.start)}
                x2={Date.parse(a.end)}
                fill={color}
                fillOpacity={active ? 0.3 : 0.16}
                stroke={active ? color : 'none'}
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
              fill={palette.band}
              fillOpacity={0.18}
              isAnimationActive={false}
              connectNulls={false}
            />
          )}

          <Line
            dataKey="restored"
            stroke={palette.series}
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
            fill={palette.observed}
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
    <div className="card" style={{ margin: 0, padding: '9px 12px' }}>
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
        // Класс аномалии словом и цветом. Цвет — переменной темы, а не из
        // словаря: в тёмной теме тёмно-красный на тёмной карточке пропадает.
        <div
          className={`small text-${inside.severity === 'critical' ? 'critical' : 'suppression'}`}
          style={{ marginTop: 4 }}
        >
          {(SEVERITY[inside.severity] || {}).label} · {CAUSE[inside.cause] || inside.cause}
        </div>
      )}
    </div>
  )
}
