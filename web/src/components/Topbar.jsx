import { useEffect, useRef, useState } from 'react'

import { useTheme } from '../theme.js'
import { IconBell, IconCalendar, IconChevron, IconPin } from './icons.jsx'

// Значки переключателя темы объявлены здесь, а не в общем наборе icons.jsx:
// они нужны ровно одному элементу интерфейса и больше нигде не встречаются.
const stroke = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.7,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
}

function IconSun(props) {
  return (
    <svg viewBox="0 0 24 24" width={19} height={19} {...stroke} {...props}>
      <circle cx="12" cy="12" r="4.2" />
      <path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.2 5.2l1.4 1.4M17.4 17.4l1.4 1.4M18.8 5.2l-1.4 1.4M6.6 17.4l-1.4 1.4" />
    </svg>
  )
}

function IconMoon(props) {
  return (
    <svg viewBox="0 0 24 24" width={19} height={19} {...stroke} {...props}>
      <path d="M20 14.4A8.4 8.4 0 0 1 9.6 4a8.4 8.4 0 1 0 10.4 10.4Z" />
    </svg>
  )
}

// «Как в системе» — монитор: тот же знак, что в системных настройках, и
// объяснять его словом не приходится.
function IconSystem(props) {
  return (
    <svg viewBox="0 0 24 24" width={19} height={19} {...stroke} {...props}>
      <rect x="2.8" y="4.2" width="18.4" height="12.6" rx="2" />
      <path d="M8.5 20.2h7M12 16.8v3.4" />
    </svg>
  )
}

// Порядок и подписи режимов. Держатся списком, потому что тот же список
// рисуется и в раскрытой панели, и в значке самой кнопки.
const THEMES = [
  { key: 'light', title: 'Светлая', Icon: IconSun, hint: 'всегда светлое' },
  { key: 'dark', title: 'Тёмная', Icon: IconMoon, hint: 'всегда тёмное' },
  { key: 'system', title: 'Как в системе', Icon: IconSystem, hint: 'по настройке устройства' },
]

// Глубина истории. В макете на этом месте выбор года, но у сервиса год — не та
// величина, которой пользователь управляет: ряд строится за несколько сезонов
// назад от сегодняшнего дня. Управляемая величина — сколько сезонов собирать,
// и она напрямую влияет и на время сбора, и на то, хватит ли ядру данных на
// собственную норму поля.
export const YEAR_OPTIONS = [
  { years: 3, title: '3 сезона', hint: 'быстрее всего, нормы едва хватает' },
  { years: 5, title: '5 сезонов', hint: 'по умолчанию: норма надёжная' },
  { years: 8, title: '8 сезонов', hint: 'дольше собирается, норма устойчивее' },
  { years: 12, title: '12 сезонов', hint: 'полная история, счёт на минуты' },
]

/**
 * Шапка рабочего поля: заголовок раздела, регион, глубина истории, уведомления.
 *
 * showRegion — показывать ли поиск региона. На «Обзоре» он не нужен: там выбор
 * региона стоит в блоке быстрого старта, ровно как в макете, где в шапке
 * остаются только глубина истории и колокольчик.
 */
export default function Topbar({
  title,
  subtitle,
  showRegion = true,
  region,
  onSearchRegion,
  places,
  searching,
  onPickPlace,
  searchNote,
  years,
  onYears,
  health,
  warnings,
}) {
  const [open, setOpen] = useState(null)
  const wrap = useRef(null)
  const { mode, setMode } = useTheme()
  const themeOption = THEMES.find((item) => item.key === mode) || THEMES[2]

  // Клик мимо закрывает раскрытую панель: без этого они накладываются друг на
  // друга и остаются висеть после перехода в другой раздел.
  useEffect(() => {
    const away = (event) => {
      if (wrap.current && !wrap.current.contains(event.target)) setOpen(null)
    }
    document.addEventListener('mousedown', away)
    return () => document.removeEventListener('mousedown', away)
  }, [])

  const down = (health?.sources || []).filter((s) => s.status !== 'ok')
  const alerts = [
    ...down.map((s) => ({ tone: s.required ? 'bad' : 'warn', text: `${s.title}: ${s.consequence}` })),
    ...(warnings || []).map((w) => ({ tone: 'warn', text: w })),
  ]

  return (
    <header className="topbar" ref={wrap}>
      <div className="topbar-title">
        <h1>{title}</h1>
        {subtitle && <div className="sub">{subtitle}</div>}
      </div>

      {/* Управление держится одной группой: по отдельности элементы
          переносятся поодиночке, и колокольчик уезжает на вторую строку один,
          как будто что-то сломалось. */}
      <div className="topbar-actions">
        {showRegion && (
          <div className="pill-wrap">
            <button className="pill" onClick={() => setOpen(open === 'region' ? null : 'region')}>
              <IconPin width={18} height={18} />
              {region || 'Выбрать регион'}
              <IconChevron className="chev" />
            </button>
            {open === 'region' && (
              <div className="popover">
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
                          setOpen(null)
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
        )}

        {/* Глубина истории */}
        <div className="pill-wrap">
          <button className="pill" onClick={() => setOpen(open === 'years' ? null : 'years')}>
            <IconCalendar width={18} height={18} />
            {years} {years >= 5 ? 'сезонов' : 'сезона'}
            <IconChevron className="chev" />
          </button>
          {open === 'years' && (
            <div className="popover wide">
              <h4>Сколько сезонов собирать</h4>
              {YEAR_OPTIONS.map((option) => (
                <button
                  key={option.years}
                  className={`menu-row${option.years === years ? ' active' : ''}`}
                  onClick={() => {
                    onYears(option.years)
                    setOpen(null)
                  }}
                >
                  <div>{option.title}</div>
                  <div className="small muted">{option.hint}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Тема оформления. Три состояния, а не выключатель на два: «как в
            системе» — это не то же самое, что «светлая», и на устройстве с
            автоматическим переключением по времени суток разница видна. */}
        <div className="pill-wrap">
          <button
            className="pill icon-only"
            onClick={() => setOpen(open === 'theme' ? null : 'theme')}
            title={`Тема: ${themeOption.title.toLowerCase()}`}
            aria-label="Тема оформления"
          >
            <themeOption.Icon />
          </button>
          {open === 'theme' && (
            <div className="popover narrow">
              <h4>Тема оформления</h4>
              {THEMES.map((option) => (
                <button
                  key={option.key}
                  className={`menu-row row theme-row${option.key === mode ? ' active' : ''}`}
                  onClick={() => {
                    setMode(option.key)
                    setOpen(null)
                  }}
                >
                  <option.Icon width={17} height={17} />
                  <span>
                    <span style={{ display: 'block' }}>{option.title}</span>
                    <span className="small muted">{option.hint}</span>
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Уведомления */}
        <div className="pill-wrap">
          <button
            className={`pill icon-only${alerts.length ? ' alert' : ''}`}
            onClick={() => setOpen(open === 'alerts' ? null : 'alerts')}
            title={alerts.length ? `Замечаний: ${alerts.length}` : 'Замечаний нет'}
          >
            <IconBell width={19} height={19} />
          </button>
          {open === 'alerts' && (
            <div className="popover">
              <h4>Состояние сбора</h4>
              {alerts.length === 0 ? (
                <p className="small muted" style={{ margin: 0 }}>
                  Все источники отвечают, данные последнего разбора собраны полностью.
                </p>
              ) : (
                alerts.map((alert, index) => (
                  <div key={index} className="row" style={{ alignItems: 'flex-start', marginBottom: 10 }}>
                    <span className={`dot ${alert.tone}`} style={{ marginTop: 7 }} />
                    <span className="small">{alert.text}</span>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
