import { useEffect, useMemo, useRef, useState } from 'react'

import { FIELD_STATE, fieldState, plural } from '../dict.js'
import { IconDots, IconPlus, IconSearch } from './icons.jsx'
import Thumb from './Thumb.jsx'

/**
 * Раздел «Участки»: всё управление сохранёнными полями в одном месте.
 *
 * Добавление живёт на карте — участок нельзя завести, не показав, где он,
 * поэтому кнопка «Добавить участок» уводит туда. Здесь остальное: посмотреть,
 * переименовать, пересчитать, удалить, открыть на карте.
 *
 * Поиск и два фильтра — не украшение: список приходит отсортированным по
 * глубине худшего отклонения, и когда полей несколько десятков, найти в нём
 * конкретное поле по имени иначе нечем.
 */
export default function Fields({ summary, selectedId, onOpen, onRename, onDelete, onAnalyze, onGoMap }) {
  // Размер миниатюры не может задаваться стилями: тайлы сдвигаются на
  // вычисленное число пикселей, и масштабирование средствами CSS сдвинуло бы
  // кадр мимо поля. Поэтому широкий экран спрашиваем напрямую.
  const wide = useMedia('(min-width: 1441px)')
  const [query, setQuery] = useState('')
  const [crop, setCrop] = useState('')
  const [status, setStatus] = useState('')
  const [editing, setEditing] = useState(null)
  const [name, setName] = useState('')

  const fields = summary?.fields || []

  // Список культур строится по тому, что действительно есть у полей: заранее
  // заданный справочник культур врал бы про хозяйство, которого мы не знаем.
  const crops = useMemo(
    () => [...new Set(fields.map((f) => f.crop_type).filter(Boolean))].sort(),
    [fields],
  )

  const shown = useMemo(
    () =>
      fields.filter((field) => {
        if (query && !field.name.toLowerCase().includes(query.trim().toLowerCase())) return false
        if (crop === '—' && field.crop_type) return false
        if (crop && crop !== '—' && field.crop_type !== crop) return false
        if (status && fieldState(field.summary) !== status) return false
        return true
      }),
    [fields, query, crop, status],
  )

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
    <>
      <div className="fields-bar">
        <label className="search">
          <IconSearch />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Поиск по участкам…"
          />
        </label>

        <select className="select" value={crop} onChange={(event) => setCrop(event.target.value)}>
          <option value="">Все культуры</option>
          {crops.map((value) => (
            <option key={value} value={value}>{value}</option>
          ))}
          <option value="—">культура не указана</option>
        </select>

        <select className="select" value={status} onChange={(event) => setStatus(event.target.value)}>
          <option value="">Все статусы</option>
          {Object.entries(FIELD_STATE).map(([key, tone]) => (
            <option key={key} value={key}>{tone.label}</option>
          ))}
        </select>

        <button className="btn primary add" onClick={onGoMap}>
          <IconPlus />
          Добавить участок
        </button>
      </div>

      <div className="card table-card">
        <div className="table-wrap">
          <table className="table fields-table">
            <thead>
              <tr>
                <th>Название участка</th>
                <th className="col-crop">Культура</th>
                <th className="num">Площадь</th>
                <th>Статус</th>
                <th>Обновлено</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {shown.map((field) => {
                const state = fieldState(field.summary)
                const tone = FIELD_STATE[state]
                return (
                  <tr
                    key={field.id}
                    className={`pick${field.id === selectedId ? ' sel' : ''}`}
                    onClick={() => onOpen(field)}
                  >
                    <td>
                      <div className="who">
                        <Thumb center={field.center} areaHa={field.area_ha} size={wide ? 68 : 54} />
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
                          <div style={{ minWidth: 0 }}>
                            <div className="name">{field.name}</div>
                            <div className="small muted">
                              {/* Культура здесь появляется только в узком окне,
                                  где отдельный столбец под неё не помещается. */}
                              <span className="crop-inline">
                                {field.crop_type || 'культура не указана'} ·{' '}
                              </span>
                              {field.source === 'osm' ? 'из OpenStreetMap' : 'нарисован вручную'}
                            </div>
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="col-crop">
                      {field.crop_type || <span className="muted small">не указана</span>}
                    </td>
                    <td className="num">{field.area_ha} га</td>
                    <td>
                      <span
                        className="status"
                        style={{ background: tone.fill, color: tone.color }}
                        title={stateHint(field, state)}
                      >
                        {tone.label}
                      </span>
                    </td>
                    <td className="small muted">{ago(field.last_analyzed_at)}</td>
                    <td onClick={(event) => event.stopPropagation()}>
                      <RowMenu
                        field={field}
                        onOpen={onOpen}
                        onAnalyze={onAnalyze}
                        onDelete={onDelete}
                        onRename={() => {
                          setEditing(field.id)
                          setName(field.name)
                        }}
                      />
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>

          {shown.length === 0 && (
            <div className="empty">
              Под фильтр ничего не подошло.
              <div style={{ marginTop: 12 }}>
                <button
                  className="btn sm"
                  onClick={() => {
                    setQuery('')
                    setCrop('')
                    setStatus('')
                  }}
                >
                  Сбросить фильтры
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="table-foot">
          {shown.length === fields.length
            ? plural(fields.length, 'участок', 'участка', 'участков')
            : `${shown.length} из ${plural(fields.length, 'участка', 'участков', 'участков')}`}
        </div>
      </div>
    </>
  )
}

/** Меню строки: действия, которых слишком много, чтобы держать их кнопками. */
function RowMenu({ field, onOpen, onAnalyze, onRename, onDelete }) {
  const [open, setOpen] = useState(false)
  const wrap = useRef(null)

  useEffect(() => {
    const away = (event) => {
      if (wrap.current && !wrap.current.contains(event.target)) setOpen(false)
    }
    document.addEventListener('mousedown', away)
    return () => document.removeEventListener('mousedown', away)
  }, [])

  const act = (fn) => () => {
    setOpen(false)
    fn()
  }

  return (
    <div className="row-menu" ref={wrap}>
      <button className="dots" title="Действия" onClick={() => setOpen(!open)}>
        <IconDots />
      </button>
      {open && (
        <div className="popover" style={{ width: 210, top: 40 }}>
          <button className="menu-row" onClick={act(() => onOpen(field))}>
            Открыть на карте
          </button>
          <button className="menu-row" onClick={act(() => onAnalyze(field))}>
            {field.summary ? 'Пересчитать' : 'Проанализировать'}
          </button>
          <button className="menu-row" onClick={act(onRename)}>
            Переименовать
          </button>
          <button className="menu-row danger" onClick={act(() => onDelete(field))}>
            Удалить
          </button>
        </div>
      )}
    </div>
  )
}

/** Следит за медиавыражением: нужен там, где размер нельзя отдать стилям. */
function useMedia(query) {
  const [matches, setMatches] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(query).matches,
  )
  useEffect(() => {
    const mq = window.matchMedia(query)
    const on = () => setMatches(mq.matches)
    on()
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [query])
  return matches
}

/**
 * Подсказка к статусу: почему поле оказалось в этом состоянии.
 *
 * Сам класс считает `fieldState` из общего словаря — тот же, которым карта
 * красит контуры. Держать здесь вторую таблицу состояний значило бы получить
 * поле, красное на карте и зелёное в списке.
 */
function stateHint(field, state) {
  const digest = field.summary
  if (!digest) return 'Поле сохранено, но ещё не разбиралось'
  if (state === 'nodata') return 'Данных не хватает, чтобы судить о состоянии сегодня'
  const current = digest.current
  if (current?.as_of) {
    const when = new Date(current.as_of).toLocaleDateString('ru-RU', { day: '2-digit', month: 'short' })
    const z = current.zscore == null ? '' : `, отклонение ${current.zscore.toFixed(1)} σ`
    return `Состояние на ${when}${z} · за всю историю периодов: ${digest.anomalies}`
  }
  if (state === 'bad') return `Критических периодов за всю историю: ${digest.critical}`
  if (state === 'watch') return `Периодов угнетения за всю историю: ${digest.suppression}`
  return 'Отклонений от нормы не найдено'
}

/** «2 ч назад» — насколько свежий разбор, без разглядывания даты. */
function ago(stamp) {
  if (!stamp) return 'не считался'
  const minutes = Math.max(0, Math.round((Date.now() - Date.parse(stamp)) / 60000))
  if (minutes < 1) return 'только что'
  if (minutes < 60) return `${plural(minutes, 'минуту', 'минуты', 'минут')} назад`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} ч назад`
  const days = Math.round(hours / 24)
  if (days < 30) return `${plural(days, 'день', 'дня', 'дней')} назад`
  return new Date(stamp).toLocaleDateString('ru-RU', { day: '2-digit', month: 'short' })
}
