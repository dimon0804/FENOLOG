import { useEffect, useRef, useState } from 'react'

import {
  IconAnalytics,
  IconChevron,
  IconFields,
  IconMap,
  IconOverview,
  IconReports,
  IconUser,
  Logo,
} from './icons.jsx'

/**
 * Тёмная навигационная колонка.
 *
 * Разделов пять, и за каждым стоит работающий экран. В макете есть шестой,
 * «Операции на полях», — его здесь нет намеренно: сервис не ведёт учёт полевых
 * работ и данных для этого раздела не существует. Пустой пункт меню на защите
 * читается как недоделка, а не как задел.
 *
 * Ниже — состояние источников и карточка пользователя, обе из макета. Имя в
 * карточке не выдумано: входа в сервисе нет, и вместо несуществующей учётной
 * записи там написано, в каком режиме сервис работает. Раскрытая карточка
 * показывает версию и позволяет перепроверить источники, не дожидаясь, пока
 * истечёт минутный кэш.
 */

export const SECTIONS = [
  { key: 'overview', title: 'Обзор', Icon: IconOverview },
  { key: 'map', title: 'Карта', Icon: IconMap },
  { key: 'fields', title: 'Участки', Icon: IconFields },
  { key: 'analytics', title: 'Аналитика', Icon: IconAnalytics },
  { key: 'reports', title: 'Отчёты', Icon: IconReports },
]

export default function Sidebar({ section, onSection, health, fieldsCount, version, onRecheck }) {
  const tone = { ok: 'ok', degraded: 'warn', down: 'bad' }[health?.status] || 'warn'
  const stateText = { ok: 'Активны', degraded: 'Частично', down: 'Сбой' }[health?.status] || '…'
  const checked = clock(health?.checked_at)

  return (
    <nav className="nav">
      <div className="nav-logo">
        <Logo />
        <div className="wordmark">
          FENO<span>LOG</span>
        </div>
      </div>

      <div className="nav-items">
        {SECTIONS.map(({ key, title, Icon }) => (
          <button
            key={key}
            className={`nav-item${section === key ? ' active' : ''}`}
            onClick={() => onSection(key)}
          >
            <Icon />
            {title}
            {key === 'fields' && fieldsCount > 0 && <span className="count">{fieldsCount}</span>}
          </button>
        ))}
      </div>

      <div className="nav-spacer" />

      <div className="nav-card">
        <div className="head">
          <b>Источники данных</b>
          <span className="state">
            <span className={`dot ${tone}`} />
            {stateText}
          </span>
        </div>
        {(health?.sources || []).map((source) => (
          <div key={source.key} className={`source-row${source.status === 'ok' ? '' : ' down'}`}>
            <div className="name">{SOURCE_TITLES[source.key] || source.title}</div>
            <div className="when">
              {source.status === 'ok'
                ? checked
                  ? `Обновлено: ${checked}`
                  : 'Отвечает'
                : 'Недоступен'}
            </div>
          </div>
        ))}
      </div>

      <Account version={version} health={health} onRecheck={onRecheck} />
    </nav>
  )
}

/** Карточка пользователя внизу колонки — место из макета. */
function Account({ version, health, onRecheck }) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const wrap = useRef(null)

  useEffect(() => {
    const away = (event) => {
      if (wrap.current && !wrap.current.contains(event.target)) setOpen(false)
    }
    document.addEventListener('mousedown', away)
    return () => document.removeEventListener('mousedown', away)
  }, [])

  async function recheck() {
    setBusy(true)
    try {
      await onRecheck?.()
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="nav-foot-wrap" ref={wrap}>
      {open && (
        <div className="nav-menu">
          <div className="small muted">
            Входа в сервисе нет: сервис открыт целиком и работает без учётных записей.
          </div>
          <div className="nav-menu-row">
            <span>Версия</span>
            <b>Фенолог {version}</b>
          </div>
          <div className="nav-menu-row">
            <span>Источники</span>
            <b>{health?.status === 'ok' ? 'все отвечают' : 'часть молчит'}</b>
          </div>
          <button className="btn sm" style={{ width: '100%' }} disabled={busy} onClick={recheck}>
            {busy ? 'Проверяю…' : 'Проверить источники заново'}
          </button>
        </div>
      )}

      <button className={`nav-foot${open ? ' open' : ''}`} onClick={() => setOpen(!open)}>
        <span className="mark">
          <IconUser width={18} height={18} />
        </span>
        <span className="who-wrap">
          <span className="who">Агроном</span>
          <span className="role">Демонстрационный режим</span>
        </span>
        <IconChevron className="chev" />
      </button>
    </div>
  )
}

/** Время последней проверки источников — коротко, как в макете: «8:30». */
function clock(stamp) {
  if (!stamp) return null
  const when = new Date(stamp)
  if (Number.isNaN(when.getTime())) return null
  return when.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
}

// Короткие имена: в узкой колонке «Planetary Computer: Sentinel-2, Landsat,
// MODIS» переносится на три строки и превращает список в стену текста.
const SOURCE_TITLES = {
  satellite: 'Planetary Computer',
  weather: 'Open-Meteo',
  parcels: 'OpenStreetMap',
  geocoder: 'Nominatim',
}
