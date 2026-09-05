import { GlyphChart, GlyphCloud, GlyphSpike, IconArrow } from './icons.jsx'
import { SEVERITY, plural } from '../dict.js'

/**
 * Экран «Обзор» — то, что видно при открытии сервиса.
 *
 * Три показателя и приглашение начать. Числа настоящие: участки и аномалии
 * берутся из сводки по сохранённым разборам, свежесть — из времени последнего
 * анализа. Придуманных значений здесь нет ни одного, иначе на защите первый же
 * вопрос «а откуда 312?» обнуляет доверие ко всему остальному.
 */
export default function Overview({ summary, onGoMap }) {
  const fields = summary?.fields || []
  const analyzed = summary?.analyzed || 0
  const anomalies = summary?.anomalies || { total: 0, critical: 0, suppression: 0 }
  const fresh = freshness(summary?.last_analyzed_at)

  return (
    <>
      <div className="stat-grid">
        <Stat
          color="var(--green)"
          glyph={<GlyphSpike />}
          label="Участков"
          value={summary?.polygons ?? '—'}
          delta={
            summary?.polygons
              ? `${summary.total_area_ha.toLocaleString('ru-RU')} га под наблюдением`
              : 'Ни одного участка не сохранено'
          }
          deltaTone={summary?.polygons ? 'var(--green)' : 'var(--ink-soft)'}
        />
        <Stat
          color="var(--purple)"
          glyph={<GlyphChart />}
          label="Аномалий"
          value={analyzed ? anomalies.total : '—'}
          delta={
            analyzed
              ? `${anomalies.critical} критических, ${anomalies.suppression} угнетения`
              : 'Ни одно поле ещё не разобрано'
          }
          deltaTone={anomalies.critical ? 'var(--critical)' : 'var(--ink-soft)'}
        />
        <Stat
          color="var(--blue)"
          glyph={<GlyphCloud />}
          label="Обновление данных"
          value={fresh.value}
          word={fresh.word}
          delta={fresh.hint}
          deltaTone="var(--ink-soft)"
        />
      </div>

      <div className="quickstart">
        <h2>Быстрый старт</h2>
        <p>Выберите регион и начните анализ ваших сельхозугодий</p>
        <button className="btn primary" onClick={onGoMap}>
          Перейти к карте
          <IconArrow width={19} height={19} />
        </button>
        <Fields />
      </div>

      {fields.length > 0 && (
        <div className="card" style={{ marginTop: 20 }}>
          <h3>Где хуже всего</h3>
          <p className="small muted" style={{ marginTop: -8 }}>
            Поля отсортированы по самому глубокому отклонению от нормы.
          </p>
          <div className="stack" style={{ gap: 0 }}>
            {fields.slice(0, 5).map((field) => (
              <FieldLine key={field.id} field={field} />
            ))}
          </div>
        </div>
      )}
    </>
  )
}

function Stat({ color, glyph, label, value, word, delta, deltaTone }) {
  return (
    <div className="stat">
      {/* Полоски у правого края — из макета. Цвет наследуется от карточки,
          поэтому у каждого показателя они свои. */}
      <div className="stripes" style={{ color }}>
        {[0, 1, 2, 3].map((i) => (
          <i key={i} />
        ))}
      </div>
      <div className="glyph" style={{ color }}>
        {glyph}
      </div>
      <div className="label">{label}</div>
      <div className={`value${word ? ' word' : ''}`}>{value}</div>
      <div className="delta" style={{ color: deltaTone }}>
        {delta}
      </div>
    </div>
  )
}

function FieldLine({ field }) {
  const digest = field.summary
  const worst = digest?.worst_zscore
  const tone = worst == null ? null : worst <= -2 ? SEVERITY.critical : SEVERITY.suppression
  return (
    <div className="row" style={{ padding: '11px 0', borderTop: '1px solid var(--line)' }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 600 }}>{field.name}</div>
        <div className="small muted">
          {field.area_ha} га{field.crop_type ? ` · ${field.crop_type}` : ''}
        </div>
      </div>
      {digest ? (
        <>
          <span className="small muted">
            {plural(digest.anomalies, 'период', 'периода', 'периодов')}
          </span>
          {tone && (
            <span className="tag" style={{ background: tone.soft, color: tone.color }}>
              {worst.toFixed(1)} σ
            </span>
          )}
        </>
      ) : (
        <span className="small muted">не анализировался</span>
      )}
    </div>
  )
}

/** Декоративная линия полей — правый край блока быстрого старта. */
function Fields() {
  return (
    <svg className="art" width="430" height="150" viewBox="0 0 430 150" fill="none">
      <g stroke="#4e9b36" strokeWidth="1.6" opacity="0.75">
        <ellipse cx="190" cy="96" rx="150" ry="26" />
        <ellipse cx="175" cy="112" rx="130" ry="22" />
        <ellipse cx="205" cy="80" rx="120" ry="20" />
        <ellipse cx="352" cy="118" rx="62" ry="13" />
        <path d="M300 60c0-9 7-16 16-16s16 7 16 16h-32zM316 60v14" />
        <path d="M370 84c0-7 5-12 12-12s12 5 12 12h-24zM382 84v11" />
        <path d="M150 108c0-6 4-10 10-10s10 4 10 10h-20zM160 108v9" />
      </g>
    </svg>
  )
}

function freshness(stamp) {
  if (!stamp) {
    return { value: 'Нет', word: true, hint: 'Разборов пока не было' }
  }
  const when = new Date(stamp)
  const now = new Date()
  const sameDay = when.toDateString() === now.toDateString()
  const time = when.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
  if (sameDay) return { value: 'Сегодня', word: true, hint: time }
  const days = Math.round((now - when) / 86400000)
  if (days <= 1) return { value: 'Вчера', word: true, hint: time }
  return {
    value: when.toLocaleDateString('ru-RU', { day: '2-digit', month: 'short' }),
    word: true,
    hint: `${plural(days, 'день', 'дня', 'дней')} назад, ${time}`,
  }
}
