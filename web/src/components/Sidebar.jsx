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
 * записи там написано, в каком режиме сервис работает.
 */

export const SECTIONS = [
  { key: 'overview', title: 'Обзор', Icon: IconOverview },
  { key: 'map', title: 'Карта', Icon: IconMap },
  { key: 'fields', title: 'Участки', Icon: IconFields },
  { key: 'analytics', title: 'Аналитика', Icon: IconAnalytics },
  { key: 'reports', title: 'Отчёты', Icon: IconReports },
]

export default function Sidebar({
  section, onSection, health, fieldsCount, version, onRecheck, rechecking,
}) {
  const tone = { ok: 'ok', degraded: 'warn', down: 'bad' }[health?.status] || 'warn'
  const stateText = { ok: 'Активны', degraded: 'Частично', down: 'Сбой' }[health?.status] || '…'

  // Время последней проверки. Панель, которая всегда показывает «отвечает» и
  // ничем не выдаёт, что она живая, неотличима от зелёной картинки для вида —
  // а это ровно та нечестность, которой в продукте быть не должно.
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
            {/* Последствие отказа сервер считает давно, но интерфейс его
                выбрасывал. «Недоступен» само по себе не отвечает на вопрос,
                можно ли продолжать работу, — а именно он у пользователя и
                возникает. */}
            {source.status !== 'ok' && source.consequence && (
              <div className="consequence">{source.consequence}</div>
            )}
          </div>
        ))}

        {/* Проверка вручную. Кэш живёт минуту, и после починки сети ждать её
            истечения, глядя на красный индикатор, незачем. */}
        <button
          type="button"
          className="source-recheck"
          onClick={onRecheck}
          disabled={rechecking}
          title="Опросить источники заново, минуя кэш"
        >
          {rechecking ? 'проверяю…' : 'проверить сейчас'}
        </button>
      </div>

      <Account version={version} health={health} />
    </nav>
  )
}

/** Карточка пользователя внизу колонки — место из макета. */
function Account({ version, health }) {
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
  cropland: 'ESA WorldCereal',
  geocoder: 'Nominatim',
}
