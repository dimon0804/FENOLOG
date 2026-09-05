import { useCallback, useEffect, useMemo, useState } from 'react'

import { api, pollTask } from './api.js'
import { CLIMATOLOGY, SENSOR, formatDate } from './dict.js'
import AnomalyFeed from './components/AnomalyFeed.jsx'
import MapPanel from './components/MapPanel.jsx'
import SeriesChart from './components/SeriesChart.jsx'
import Sidebar from './components/Sidebar.jsx'
import SourcesHealth from './components/SourcesHealth.jsx'
import WeatherPanel from './components/WeatherPanel.jsx'

/**
 * Один экран на весь сценарий.
 *
 * Путь пользователя: найти регион -> найти в нём готовые контуры или нарисовать
 * свой -> запустить анализ -> увидеть ряд, периоды и их объяснение -> сохранить
 * поле, чтобы вернуться к нему позже. Оба способа задать территорию сходятся в
 * одну ветку: дальше сервису всё равно, откуда взялся контур.
 */
export default function App() {
  const [polygons, setPolygons] = useState([])
  const [selected, setSelected] = useState(null)      // сохранённый участок
  const [draft, setDraft] = useState(null)            // ещё не сохранённый контур

  const [places, setPlaces] = useState([])
  const [searching, setSearching] = useState(false)
  const [searchNote, setSearchNote] = useState(null)
  const [regionOutline, setRegionOutline] = useState(null)
  const [flyTo, setFlyTo] = useState(null)

  const [parcels, setParcels] = useState([])
  const [discovering, setDiscovering] = useState(false)
  const [discoverNote, setDiscoverNote] = useState(null)

  const [task, setTask] = useState(null)
  const [result, setResult] = useState(null)
  const [activeAnomaly, setActiveAnomaly] = useState(null)
  const [error, setError] = useState(null)

  const reloadPolygons = useCallback(
    () => api.listPolygons().then((r) => setPolygons(r.polygons)).catch(() => {}),
    [],
  )
  useEffect(() => { reloadPolygons() }, [reloadPolygons])

  const geometry = draft?.geometry || selected?.geometry || null

  // ------------------------------------------------------------------- регион
  async function searchRegion(query) {
    setSearching(true)
    setSearchNote(null)
    try {
      const found = await api.searchRegion(query)
      setPlaces(found.places)
      if (!found.places.length) {
        setSearchNote(
          found.geocoder_available
            ? 'Ничего не нашлось. Попробуйте другое написание или более крупный объект.'
            : 'Поиск региона сейчас недоступен. Карту можно двигать вручную — на сценарий это не влияет.',
        )
      }
    } catch (e) {
      setSearchNote(e.message)
    } finally {
      setSearching(false)
    }
  }

  function pickPlace(place) {
    setFlyTo({ bbox: place.bbox, at: Date.now() })
    setRegionOutline(place.geometry)
    setPlaces([])
  }

  // -------------------------------------------------------------- поиск полей
  async function discover(bbox) {
    setDiscovering(true)
    setDiscoverNote(null)
    setError(null)
    try {
      const found = await api.discover(bbox)
      setParcels(found.parcels)
      setDiscoverNote(found.note)
    } catch (e) {
      setError(e.message)
    } finally {
      setDiscovering(false)
    }
  }

  function pickParcel(id) {
    const parcel = parcels.find((p) => p.id === id)
    if (!parcel) return
    setSelected(null)
    setResult(null)
    setTask(null)
    setDraft({
      geometry: parcel.geometry,
      name: parcel.name || `Контур ${parcel.id}`,
      crop_type: parcel.crop_hint,
      area_ha: parcel.area_ha,
      source: 'osm',
      external_id: parcel.id,
    })
  }

  function onDrawn(geom) {
    setSelected(null)
    setResult(null)
    setTask(null)
    setParcels([])
    setDraft({ geometry: geom, name: '', crop_type: null, source: 'drawn', external_id: null })
  }

  // ------------------------------------------------------------------- анализ
  async function runAnalysis(starter) {
    setError(null)
    setResult(null)
    setActiveAnomaly(null)
    try {
      const started = await starter()
      setTask(started.task)
      const finished = await pollTask(started.task.id, setTask)
      setTask(finished)
      setResult(finished.result)
      reloadPolygons()
    } catch (e) {
      setError(e.message)
      setTask(null)
    }
  }

  const analyzeDraft = (save) =>
    runAnalysis(async () => {
      const response = await api.analyzeGeometry({
        geometry: draft.geometry,
        crop_type: draft.crop_type,
        name: draft.name || null,
        source: draft.source,
        external_id: draft.external_id,
        save,
      })
      if (response.polygon) {
        setSelected(response.polygon)
        setDraft(null)
        await reloadPolygons()
      }
      return response
    })

  const analyzeSaved = (polygon) => runAnalysis(() => api.analyzePolygon(polygon.id, {}))

  async function selectPolygon(polygon) {
    setSelected(polygon)
    setDraft(null)
    setParcels([])
    setActiveAnomaly(null)
    setTask(null)
    setError(null)
    setFlyTo({ bbox: bboxOf(polygon.geometry), at: Date.now() })
    // Прошлый анализ показывается сразу и без сети до самого конца: возвращаться
    // к полю и каждый раз ждать минуту сбора — ровно то, ради чего результат и
    // кладётся на диск. У поля, которое ещё не считали, результата заведомо нет —
    // не спрашиваем, чтобы не сорить четырёхсотыми в консоли.
    if (!polygon.last_analyzed_at) {
      setResult(null)
      return
    }
    try {
      const saved = await api.savedResult(polygon.id)
      setResult(saved.result)
    } catch {
      setResult(null)
    }
  }

  async function renamePolygon(id, name) {
    await api.renamePolygon(id, name)
    await reloadPolygons()
    if (selected?.id === id) setSelected({ ...selected, name })
  }

  async function deletePolygon(polygon) {
    if (!window.confirm(`Удалить участок «${polygon.name}»?`)) return
    await api.deletePolygon(polygon.id)
    if (selected?.id === polygon.id) {
      setSelected(null)
      setResult(null)
    }
    reloadPolygons()
  }

  const anomalies = result?.anomalies || []
  // Нарисованный контур имени ещё не имеет, но он уже выбран — заголовок
  // «Поле не выбрано» над строкой «нарисован вручную» противоречит сам себе.
  const title =
    selected?.name || draft?.name || (draft ? 'Новый контур' : 'Поле не выбрано')

  return (
    <div className="app">
      <header className="topbar">
        <h1>Фенолог</h1>
        <span className="subtitle">мониторинг вегетационной динамики</span>
        <span className="spacer" />
        <SourcesHealth />
      </header>

      <div className="workspace">
        <Sidebar
          polygons={polygons}
          selectedId={selected?.id}
          onSelect={selectPolygon}
          onRename={renamePolygon}
          onDelete={deletePolygon}
          onSearch={searchRegion}
          places={places}
          searching={searching}
          onPickPlace={pickPlace}
          searchNote={searchNote}
        />

        <div className="stage">
          <MapPanel
            parcels={parcels}
            selectedGeometry={geometry}
            regionOutline={regionOutline}
            flyTo={flyTo}
            onPickParcel={pickParcel}
            onDrawn={onDrawn}
            onDiscover={discover}
            discovering={discovering}
            discoverNote={discoverNote}
          />

          <div className="analysis">
            <div className="charts">
              <FieldHeader
                title={title}
                draft={draft}
                selected={selected}
                result={result}
                task={task}
                error={error}
                onAnalyzeDraft={analyzeDraft}
                onAnalyzeSaved={analyzeSaved}
                onDraftName={(name) => setDraft({ ...draft, name })}
              />

              {result && (
                <>
                  <SeriesChart
                    series={result.series}
                    anomalies={anomalies}
                    activeAnomaly={activeAnomaly}
                    onPickAnomaly={setActiveAnomaly}
                  />
                  <WeatherPanel
                    weather={result.weather}
                    anomalies={anomalies}
                    activeAnomaly={activeAnomaly}
                  />
                </>
              )}
            </div>

            <div className="feed">
              {result ? (
                <AnomalyFeed
                  anomalies={anomalies}
                  active={activeAnomaly}
                  onPick={(index) => setActiveAnomaly(activeAnomaly === index ? null : index)}
                  climatologySource={result.meta?.climatology_source}
                />
              ) : (
                <div className="empty">
                  Здесь появятся негативные аномальные периоды и объяснение их причины.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/** Шапка панели поля: что выбрано, чем это считалось и кнопка запуска. */
function FieldHeader({ title, draft, selected, result, task, error, onAnalyzeDraft, onAnalyzeSaved, onDraftName }) {
  const running = task && task.status !== 'done' && task.status !== 'failed'
  const meta = result?.meta || {}
  const climatology = CLIMATOLOGY[meta.climatology_source]

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
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
                placeholder="название поля"
                value={draft.name}
                style={{ width: 190 }}
                onChange={(event) => onDraftName(event.target.value)}
              />
              <button onClick={() => onAnalyzeDraft(false)} disabled={running}>
                Только посчитать
              </button>
              <button className="primary" onClick={() => onAnalyzeDraft(true)} disabled={running}>
                Сохранить и посчитать
              </button>
            </>
          )}
          {selected && !draft && (
            <button className="primary" onClick={() => onAnalyzeSaved(selected)} disabled={running}>
              {result ? 'Пересчитать' : 'Проанализировать'}
            </button>
          )}
        </div>
      </div>

      {running && (
        <div style={{ marginTop: 12 }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <span className="small">{task.stage}</span>
            <span className="small muted">{task.percent}%</span>
          </div>
          <div className="progress"><div style={{ width: `${task.percent}%` }} /></div>
        </div>
      )}

      {error && <div className="error-banner" style={{ marginTop: 12 }}>{error}</div>}

      {task?.warnings?.length > 0 && (
        <div className="error-banner" style={{ marginTop: 12 }}>
          Часть данных собрать не удалось, разбор построен на остальных:{' '}
          {task.warnings.join('; ')}
        </div>
      )}

      {result && (
        <div className="row" style={{ marginTop: 12, flexWrap: 'wrap', gap: 10 }}>
          {climatology && (
            <span className={`badge ${climatology.tone === 'ok' ? '' : climatology.tone}`} title={climatology.hint}>
              <span className={`dot ${climatology.tone}`} />
              {climatology.label}
            </span>
          )}
          <span className="badge">
            {meta.collected_observations} наблюдений
            {meta.sources && Object.keys(meta.sources).length > 0 && (
              <> · {Object.entries(meta.sources).map(([k, v]) => `${SENSOR[k] || k}: ${v}`).join(', ')}</>
            )}
          </span>
          <span className="badge">погода: {meta.collected_weather_days} дней</span>
          <span className="badge">
            {formatDate(meta.date_from)} — {formatDate(meta.date_to)}
          </span>
          <span className="badge">сбор {meta.collect_seconds} с</span>
        </div>
      )}

      {result && climatology && climatology.tone !== 'ok' && (
        <p className="small muted" style={{ marginBottom: 0 }}>{climatology.hint}</p>
      )}
    </div>
  )
}

/** Рамка контура — нужна, чтобы карта перелетела к выбранному участку. */
function bboxOf(geometry) {
  const rings =
    geometry.type === 'Polygon' ? geometry.coordinates : geometry.coordinates.flat()
  let west = 180, south = 90, east = -180, north = -90
  for (const ring of rings) {
    for (const [lon, lat] of ring) {
      if (lon < west) west = lon
      if (lon > east) east = lon
      if (lat < south) south = lat
      if (lat > north) north = lat
    }
  }
  return [west, south, east, north]
}
