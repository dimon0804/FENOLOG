import { CLIMATOLOGY, SENSOR, formatDate } from '../dict.js'

/**
 * Карточка выбранного поля: что выбрано, чем это считалось, кнопка запуска и
 * прогресс сбора.
 *
 * Значок источника нормы вынесен сюда отдельно и с подсказкой: пользователь
 * должен понимать, когда норма построена по истории самого поля, а когда это
 * прикидка по культуре. Ровно та честность, за которую дают баллы за продукт.
 */
export default function FieldPanel({
  draft,
  selected,
  result,
  task,
  error,
  years,
  onAnalyzeDraft,
  onAnalyzeSaved,
  onDraftName,
}) {
  const running = task && task.status !== 'done' && task.status !== 'failed'
  const meta = result?.meta || {}
  const climatology = CLIMATOLOGY[meta.climatology_source]
  // Нарисованный контур уже выбран, но имени у него ещё нет: заголовок
  // «Поле не выбрано» над строкой «нарисован вручную» противоречил бы сам себе.
  const title = selected?.name || draft?.name || (draft ? 'Новый контур' : 'Поле не выбрано')

  return (
    <div className="card">
      <div className="row wrap" style={{ justifyContent: 'space-between' }}>
        <div>
          <h3 style={{ marginBottom: 2 }}>{title}</h3>
          <div className="small muted">
            {draft
              ? `${draft.source === 'osm' ? 'контур из OpenStreetMap' : 'нарисован вручную'}${
                  draft.area_ha ? ` · ${draft.area_ha} га` : ''
                }`
              : selected
                ? `${selected.area_ha} га${selected.crop_type ? ` · ${selected.crop_type}` : ''}`
                : 'Найдите поле на карте или нарисуйте контур'}
          </div>
        </div>

        <div className="row">
          {draft && (
            <>
              <input
                type="text"
                className="field"
                style={{ width: 200 }}
                placeholder="название поля"
                value={draft.name}
                onChange={(event) => onDraftName(event.target.value)}
              />
              <button className="btn" onClick={() => onAnalyzeDraft(false)} disabled={running}>
                Только посчитать
              </button>
              <button className="btn primary" onClick={() => onAnalyzeDraft(true)} disabled={running}>
                Сохранить и посчитать
              </button>
            </>
          )}
          {selected && !draft && result && (
            // Отчёт для тех, кто не читает графики: обычная ссылка, а не fetch,
            // потому что файл должен уйти в загрузки браузера, а не в память
            // вкладки. Имя файла сервер задаёт сам заголовком.
            <a
              className="btn"
              href={`/api/polygons/${selected.id}/report.pdf`}
              download
              title="PDF с графиками и объяснением обычными словами"
            >
              Отчёт PDF
            </a>
          )}
          {selected && !draft && (
            <button className="btn primary" onClick={() => onAnalyzeSaved(selected)} disabled={running}>
              {result ? 'Пересчитать' : 'Проанализировать'}
            </button>
          )}
        </div>
      </div>

      {running && (
        <div style={{ marginTop: 14 }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <span className="small">{task.stage}</span>
            <span className="small muted">{task.percent}%</span>
          </div>
          <div className="progress">
            <div style={{ width: `${task.percent}%` }} />
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>
            Собираем {years} сезонов истории. Первый разбор поля идёт от полуминуты до пары
            минут, повторное открытие — мгновенно.
          </div>
        </div>
      )}

      {error && <div className="banner" style={{ marginTop: 14 }}>{error}</div>}

      {task?.warnings?.length > 0 && (
        <div className="banner warn" style={{ marginTop: 14 }}>
          Часть данных собрать не удалось, разбор построен на остальных: {task.warnings.join('; ')}
        </div>
      )}

      {result && (
        <div className="row wrap" style={{ marginTop: 14, gap: 10 }}>
          {climatology && (
            <span className={`chip ${climatology.tone === 'ok' ? '' : climatology.tone}`} title={climatology.hint}>
              <span className={`dot ${climatology.tone}`} />
              {climatology.label}
            </span>
          )}
          <span className="chip">
            {meta.collected_observations} наблюдений
            {meta.sources && Object.keys(meta.sources).length > 0 && (
              <>
                {' · '}
                {Object.entries(meta.sources)
                  .map(([key, value]) => `${SENSOR[key] || key}: ${value}`)
                  .join(', ')}
              </>
            )}
          </span>
          <span className="chip">погода: {meta.collected_weather_days} дней</span>
          <span className="chip">
            {formatDate(meta.date_from)} — {formatDate(meta.date_to)}
          </span>
          <span className="chip">сбор {meta.collect_seconds} с</span>
        </div>
      )}

      {result && climatology && climatology.tone !== 'ok' && (
        <p className="small muted" style={{ marginBottom: 0 }}>{climatology.hint}</p>
      )}
    </div>
  )
}
