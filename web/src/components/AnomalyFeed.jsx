import { CAUSE, SEVERITY, formatDate, plural } from '../dict.js'

/**
 * Лента найденных периодов.
 *
 * Пятый пункт обязательного минимума постановки — интерпретация аномалий.
 * Фраза приходит из ядра готовой и выводится как есть: ядро видело и погоду, и
 * пороги, и окно уборки для культуры, интерфейс — нет, и переписывать её здесь
 * значило бы врать точнее источника.
 */
export default function AnomalyFeed({ anomalies, active, onPick, climatologySource }) {
  if (!anomalies?.length) {
    return (
      <div className="empty">
        {climatologySource === 'none' ? (
          <>
            Периоды не искались: у поля нет ни собственной истории, ни нормы по культуре —
            сравнивать не с чем.
          </>
        ) : (
          <>Негативных отклонений от нормы не найдено — поле развивалось штатно.</>
        )}
      </div>
    )
  }

  return (
    <div>
      {anomalies.map((anomaly, index) => {
        const tone = SEVERITY[anomaly.severity] || SEVERITY.suppression
        return (
          <div
            key={`${anomaly.start}-${anomaly.end}`}
            className={`anomaly${active === index ? ' active' : ''}`}
            onClick={() => onPick?.(index)}
          >
            <div className="head">
              <span className="dates">
                {formatDate(anomaly.start)} — {formatDate(anomaly.end)}
              </span>
              <span
                className="severity-tag"
                style={{ background: tone.soft, color: tone.color }}
              >
                {tone.label}
              </span>
            </div>

            <div className="cause">
              {CAUSE[anomaly.cause] || anomaly.cause}
              {anomaly.cause_confidence > 0 && (
                <> · уверенность {Math.round(anomaly.cause_confidence * 100)}%</>
              )}
            </div>

            <div className="explanation">{anomaly.explanation}</div>

            <div className="numbers">
              <span>{plural(anomaly.duration_days, 'день', 'дня', 'дней')}</span>
              <span>минимум {anomaly.min_zscore?.toFixed(1)} σ</span>
              <span>в среднем {anomaly.mean_zscore?.toFixed(1)} σ</span>
            </div>

            {active === index && <Evidence evidence={anomaly.evidence} />}
          </div>
        )
      })}
    </div>
  )
}

// Числа, на которых ядро построило версию причины. Показываются только у
// раскрытого периода: в свёрнутом виде они перегружают ленту, а в раскрытом
// отвечают на вопрос «почему сервис так решил».
const EVIDENCE_LABELS = {
  precip_30d_mm: ['Осадки за 30 дней', 'мм'],
  precip_30d_norm_mm: ['Норма осадков за 30 дней', 'мм'],
  precip_ratio: ['Доля от нормы осадков', ''],
  temp_mean_c: ['Средняя температура', '°C'],
  temp_norm_c: ['Норма температуры', '°C'],
  temp_min_c: ['Минимальная температура', '°C'],
  temp_anomaly_c: ['Отклонение температуры', '°C'],
  z_drop_10d: ['Обвал индекса за 10 дней', 'σ'],
  start_doy: ['День года начала', ''],
  harvest_window: ['Характерное окно уборки', 'дни года'],
  norm_source: ['Источник нормы', ''],
}

function Evidence({ evidence }) {
  const rows = Object.entries(evidence || {}).filter(([key]) => EVIDENCE_LABELS[key])
  if (!rows.length) return null
  return (
    <div className="numbers" style={{ flexDirection: 'column', gap: 2, marginTop: 8 }}>
      {rows.map(([key, value]) => {
        const [label, unit] = EVIDENCE_LABELS[key]
        const text = Array.isArray(value) ? value.join('—') : value
        return (
          <span key={key}>
            {label}: {text} {unit}
          </span>
        )
      })}
    </div>
  )
}
