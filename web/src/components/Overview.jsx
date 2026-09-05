import { useEffect, useRef, useState } from 'react'

import { GlyphChart, GlyphCloud, GlyphSpike, IconArrow, IconChevron } from './icons.jsx'
import { SEVERITY, plural } from '../dict.js'

/**
 * Экран «Обзор» — то, что видно при открытии сервиса.
 *
 * Верхний ряд — три показателя, нижний — быстрый старт и недавняя активность
 * рядом, как в макете. Числа настоящие: участки и аномалии берутся из сводки по
 * сохранённым разборам, свежесть — из времени последнего анализа. Придуманных
 * значений здесь нет ни одного, иначе на защите первый же вопрос «а откуда 312?»
 * обнуляет доверие ко всему остальному.
 */
export default function Overview({
  summary,
  onGoMap,
  onOpenField,
  region,
  onSearchRegion,
  places,
  searching,
  searchNote,
  onPickPlace,
}) {
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

      {/* Нижний ряд макета: приглашение начать слева, что происходило — справа. */}
      <div className="overview-bottom">
        <QuickStart
          region={region}
          onSearchRegion={onSearchRegion}
          places={places}
          searching={searching}
          searchNote={searchNote}
          onPickPlace={onPickPlace}
          onGoMap={onGoMap}
        />
        <Activity summary={summary} onOpenField={onOpenField} onGoMap={onGoMap} />
      </div>
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

/**
 * Быстрый старт: выбрать регион и уйти на карту.
 *
 * Выбор региона стоит здесь, а не только в шапке, потому что это первое
 * действие сценария: без района карта открывается над пустым местом, и человек
 * не понимает, что делать дальше. Список районов не зашит — он приходит от
 * геокодера, поэтому подходит любое название, а не десяток заготовленных.
 */
function QuickStart({ region, onSearchRegion, places, searching, searchNote, onPickPlace, onGoMap }) {
  const [open, setOpen] = useState(false)
  const wrap = useRef(null)

  useEffect(() => {
    const away = (event) => {
      if (wrap.current && !wrap.current.contains(event.target)) setOpen(false)
    }
    document.addEventListener('mousedown', away)
    return () => document.removeEventListener('mousedown', away)
  }, [])

  return (
    <section className="quickstart">
      <h2>Быстрый старт</h2>
      <p>Выберите регион и начните анализ ваших сельхозугодий</p>

      <div className="pick" ref={wrap}>
        <button className="pick-btn" onClick={() => setOpen(!open)}>
          <span className={region ? '' : 'ph'}>{region || 'Выберите регион'}</span>
          <IconChevron className="chev" />
        </button>
        {open && (
          <div className="popover left" style={{ top: 50 }}>
            <h4>Поиск региона</h4>
            <form
              className="stack"
              onSubmit={(event) => {
                event.preventDefault()
                onSearchRegion(new FormData(event.currentTarget).get('q'))
              }}
            >
              <input
                type="search"
                className="field"
                name="q"
                autoFocus
                placeholder="Сальский район, Кубань, Аксай…"
              />
              <button className="btn primary" type="submit" disabled={searching}>
                {searching ? 'Ищу…' : 'Найти на карте'}
              </button>
            </form>
            {searchNote && <p className="small muted">{searchNote}</p>}
            {places?.length > 0 && (
              <div style={{ marginTop: 10 }}>
                {places.map((place) => (
                  <button
                    key={`${place.name}-${place.center.join()}`}
                    className="menu-row"
                    onClick={() => {
                      onPickPlace(place)
                      setOpen(false)
                    }}
                  >
                    <div>{place.name.split(',').slice(0, 3).join(',')}</div>
                    <div className="small muted">{place.type}</div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <button className="btn go" onClick={onGoMap}>
        Перейти к карте
        <IconArrow width={19} height={19} />
      </button>
    </section>
  )
}

/**
 * Недавняя активность: что происходило с полями и когда.
 *
 * Порядок — по времени последнего разбора, а не по глубине аномалии: экран
 * отвечает на вопрос «что нового», и свежий пересчёт спокойного поля здесь
 * важнее давно известной беды. Насколько всё плохо, видно по точке слева и по
 * метке отклонения справа.
 */
function Activity({ summary, onOpenField, onGoMap }) {
  const fields = [...(summary?.fields || [])].sort(
    (a, b) => Date.parse(b.last_analyzed_at || 0) - Date.parse(a.last_analyzed_at || 0),
  )

  return (
    <section className="activity">
      <h2>Недавняя активность</h2>

      {fields.length === 0 ? (
        <div className="empty">
          Здесь появятся разобранные поля.
          <div style={{ marginTop: 14 }}>
            <button className="btn primary" onClick={onGoMap}>
              Выбрать поле на карте
            </button>
          </div>
        </div>
      ) : (
        <div className="activity-list">
          {fields.slice(0, 6).map((field) => (
            <ActivityRow key={field.id} field={field} onOpen={onOpenField} />
          ))}
        </div>
      )}
    </section>
  )
}

function ActivityRow({ field, onOpen }) {
  const digest = field.summary
  const worst = digest?.worst_zscore
  const tone =
    !digest ? null : worst != null && worst <= -2 ? SEVERITY.critical : SEVERITY.suppression

  const state = rowState(digest)

  return (
    <button className="activity-row" onClick={() => onOpen(field)}>
      <span className={`dot ${state.dot}`} />
      <span className="who">
        <span className="name">{field.name}</span>
        <span className="what">{state.text}</span>
      </span>
      {digest && worst != null && (
        <span className="tag" style={{ background: tone.soft, color: tone.color }}>
          {worst.toFixed(1)} σ
        </span>
      )}
      <span className="when">{ago(field.last_analyzed_at)}</span>
    </button>
  )
}

/** Точка и подпись строки: что именно случилось с полем при последнем разборе. */
function rowState(digest) {
  if (!digest) return { dot: 'idle', text: 'Ещё не разбиралось' }
  if (digest.critical > 0) {
    return {
      dot: 'bad',
      text: `Критических аномалий: ${digest.critical}`,
    }
  }
  if (digest.anomalies > 0) {
    return {
      dot: 'warn',
      text: `${plural(digest.anomalies, 'период', 'периода', 'периодов')} угнетения`,
    }
  }
  return { dot: 'ok', text: 'Данные обновлены, отклонений нет' }
}

/** «2 ч назад» — насколько свежий разбор, без разглядывания даты. */
function ago(stamp) {
  if (!stamp) return 'не разбиралось'
  const minutes = Math.max(0, Math.round((Date.now() - Date.parse(stamp)) / 60000))
  if (minutes < 1) return 'только что'
  if (minutes < 60) return `${plural(minutes, 'минуту', 'минуты', 'минут')} назад`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} ч назад`
  const days = Math.round(hours / 24)
  if (days < 30) return `${plural(days, 'день', 'дня', 'дней')} назад`
  return new Date(stamp).toLocaleDateString('ru-RU', { day: '2-digit', month: 'short' })
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
