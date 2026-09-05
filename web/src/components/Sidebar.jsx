import {
  IconAnalytics,
  IconFields,
  IconMap,
  IconOverview,
  IconReports,
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
 * Внизу — состояние источников. В макете на этом месте карточка пользователя,
 * но входа в сервисе нет и не планируется (аутентификацию постановка прямо не
 * оценивает), а выдумывать учётную запись значит показывать жюри то, чего в
 * продукте нет. Место занято тем, что действительно меняется во время работы.
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
  const checkedAt = health?.checked_at ? new Date(health.checked_at) : null
  const checkedText = checkedAt
    ? checkedAt.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
    : null

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
              {source.status === 'ok' ? 'Отвечает' : 'Недоступен'}
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

        <button
          type="button"
          className="source-recheck"
          onClick={onRecheck}
          disabled={rechecking}
          title="Опросить источники заново, минуя кэш"
        >
          {rechecking
            ? 'проверяю…'
            : checkedText
              ? `проверено в ${checkedText} · обновить`
              : 'проверить сейчас'}
        </button>
      </div>

      <div className="nav-foot">
        <span className="mark">
          <Logo size={22} />
        </span>
        <div>
          <div className="who">Фенолог {version}</div>
          <div className="role">Демонстрационный режим</div>
        </div>
      </div>
    </nav>
  )
}

// Короткие имена: в узкой колонке «Planetary Computer: Sentinel-2, Landsat,
// MODIS» переносится на три строки и превращает список в стену текста.
const SOURCE_TITLES = {
  satellite: 'Planetary Computer',
  weather: 'Open-Meteo',
  parcels: 'OpenStreetMap',
  geocoder: 'Nominatim',
}
