import { useCallback, useEffect, useMemo, useState } from 'react'

import { api, pollTask } from './api.js'
import { REGION_PRESETS, fieldState } from './dict.js'
import Analytics from './components/Analytics.jsx'
import Fields from './components/Fields.jsx'
import MapWorkspace from './components/MapWorkspace.jsx'
import Overview from './components/Overview.jsx'
import Reports from './components/Reports.jsx'
import Sidebar from './components/Sidebar.jsx'
import Topbar from './components/Topbar.jsx'

const VERSION = '1.0'

// Заголовок и подзаголовок каждого раздела. Держим в одном месте: они попадают
// и в шапку, и в логику раздела, и расхождение между ними сразу заметно.
const TITLES = {
  overview: ['Добро пожаловать', 'Независимые данные и прогнозы для вашего агробизнеса'],
  map: ['Карта', 'Найдите готовый контур или нарисуйте свой — дальше сервис всё сделает сам'],
  fields: ['Участки', 'Сохранённые поля: пересчёт, переименование, удаление'],
  analytics: ['Аналитика', 'Сводка по хозяйству и сравнение полей между собой'],
  reports: ['Отчёты', 'Выгрузка ряда и найденных периодов файлом'],
}

/**
 * Оболочка приложения: тёмная колонка разделов, шапка и рабочее поле.
 *
 * Состояние держится здесь целиком, а не по разделам: выбранное поле и его
 * разбор нужны и карте, и аналитике, и отчётам. Разносить его по экранам
 * значило бы пересчитывать одно и то же при каждом переходе.
 */
export default function App() {
  const [section, setSection] = useState('overview')

  const [summary, setSummary] = useState(null)
  const [health, setHealth] = useState(null)

  const [selected, setSelected] = useState(null) // сохранённый участок
  const [draft, setDraft] = useState(null) // ещё не сохранённый контур
  const [result, setResult] = useState(null)
  const [activeAnomaly, setActiveAnomaly] = useState(null)
  const [task, setTask] = useState(null)
  const [error, setError] = useState(null)

  const [places, setPlaces] = useState([])
  const [searching, setSearching] = useState(false)
  const [searchNote, setSearchNote] = useState(null)
  const [region, setRegion] = useState(null)
  // Ключ пресета Южного федерального округа, если регион выбран выпадающим
  // списком на карте. У региона из поиска ключа нет — и список не должен
  // показывать чужое название вместо найденного.
  const [regionKey, setRegionKey] = useState(null)
  const [regionOutline, setRegionOutline] = useState(null)
  const [flyTo, setFlyTo] = useState(null)

  // Сохранённые контуры с геометрией — карта рисует их поверх подложки и
  // красит по состоянию. В сводке геометрии нет (там только центр), поэтому
  // список участков приходит отдельным маршрутом.
  const [polygons, setPolygons] = useState([])

  const [parcels, setParcels] = useState([])
  const [discovering, setDiscovering] = useState(false)
  const [discoverNote, setDiscoverNote] = useState(null)

  const [years, setYears] = useState(5)

  // Сводка и список участков грузятся вместе: в сводке есть вердикт по каждому
  // полю, но нет геометрии, а карте нужно и то и другое. Падение одного из
  // запросов не должно ронять второй — отсюда allSettled.
  const reload = useCallback(async () => {
    const [digest, list] = await Promise.allSettled([api.summary(), api.listPolygons()])
    if (digest.status === 'fulfilled') setSummary(digest.value)
    if (list.status === 'fulfilled') setPolygons(list.value.polygons || [])
  }, [])
  useEffect(() => { reload() }, [reload])

  // Состояние источников опрашиваем раз в минуту: ровно столько живёт кэш
  // проверки на сервере, чаще спрашивать бессмысленно.
  useEffect(() => {
    let alive = true
    const load = () => api.health().then((h) => alive && setHealth(h)).catch(() => {})
    load()
    const timer = setInterval(load, 60000)
    return () => { alive = false; clearInterval(timer) }
  }, [])

  const geometry = draft?.geometry || selected?.geometry || null

  // Что показать в карточке над выбранным контуром: название и культура —
  // ровно две строки макета.
  const selectedLabel = draft
    ? { name: draft.name || 'Новый контур', crop: draft.crop_type }
    : selected
      ? { name: selected.name, crop: selected.crop_type }
      : null

  // Раскраска сохранённых полей на карте. Вердикт считается из выжимки разбора
  // той же логикой, что и в ядре; поле без разбора — «нет данных», серое.
  const savedFields = useMemo(() => {
    const digests = new Map((summary?.fields || []).map((field) => [field.id, field.summary]))
    return polygons
      .filter((item) => item.geometry)
      .map((item) => ({
        id: item.id,
        geometry: item.geometry,
        state: fieldState(digests.get(item.id)),
      }))
  }, [polygons, summary])

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
    setRegion(place.name.split(',')[0])
    setRegionKey(null)
    setRegionOutline(place.geometry)
    setFlyTo({ bbox: place.bbox, at: Date.now() })
    setPlaces([])
    setSection('map')
  }

  /** Регион из выпадающего списка на карте: перелёт к рамке пресета. */
  function pickRegion(key) {
    const preset = REGION_PRESETS.find((item) => item.key === key)
    if (!preset) return
    const [west, south, east, north] = preset.bbox
    setRegionKey(key)
    setRegion(preset.title)
    // Пунктиром показывается именно рамка поиска, а не граница субъекта:
    // пресеты — прямоугольники, и рисовать их как настоящую границу было бы
    // враньём на карте.
    setRegionOutline({
      type: 'Polygon',
      coordinates: [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    })
    setFlyTo({ bbox: preset.bbox, at: Date.now() })
    setSection('map')
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
    setSection('map')
    try {
      const started = await starter()
      setTask(started.task)
      const finished = await pollTask(started.task.id, setTask)
      setTask(finished)
      setResult(finished.result)
      reload()
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
        years,
        save,
      })
      if (response.polygon) {
        setSelected(response.polygon)
        setDraft(null)
        await reload()
      }
      return response
    })

  const analyzeSaved = (field) =>
    runAnalysis(async () => {
      const polygon = field.geometry ? field : await api.polygon(field.id)
      setSelected(polygon)
      setDraft(null)
      return api.analyzePolygon(polygon.id, { years })
    })

  /** Открыть сохранённый участок: карта летит к нему, разбор приходит с диска. */
  async function openField(field) {
    setDraft(null)
    setParcels([])
    setActiveAnomaly(null)
    setTask(null)
    setError(null)
    try {
      const polygon = field.geometry ? field : await api.polygon(field.id)
      setSelected(polygon)
      setFlyTo({ bbox: bboxOf(polygon.geometry), at: Date.now() })
      setSection('map')
      // Поле, которое ещё не считали, результата заведомо не имеет — не
      // спрашиваем, чтобы не сорить четырёхсотыми в консоли.
      if (!polygon.last_analyzed_at) {
        setResult(null)
        return
      }
      const saved = await api.savedResult(polygon.id)
      setResult(saved.result)
    } catch {
      setResult(null)
    }
  }

  async function renameField(id, name) {
    await api.renamePolygon(id, name)
    await reload()
    if (selected?.id === id) setSelected({ ...selected, name })
  }

  async function deleteField(field) {
    if (!window.confirm(`Удалить участок «${field.name}»?`)) return
    await api.deletePolygon(field.id)
    if (selected?.id === field.id) {
      setSelected(null)
      setResult(null)
    }
    reload()
  }

  const [titleBase, subtitle] = TITLES[section]
  const title = section === 'overview' ? `${titleBase}!` : titleBase

  const shared = {
    draft, selected, result, task, error, years,
    onAnalyzeDraft: analyzeDraft,
    onAnalyzeSaved: analyzeSaved,
    onDraftName: (name) => setDraft({ ...draft, name }),
    activeAnomaly, setActiveAnomaly,
  }

  return (
    <div className="shell">
      <Sidebar
        section={section}
        onSection={setSection}
        health={health}
        fieldsCount={summary?.polygons || 0}
        version={VERSION}
      />

      <div className="main">
        <Topbar
          title={title}
          subtitle={subtitle}
          region={region}
          onSearchRegion={searchRegion}
          places={places}
          searching={searching}
          onPickPlace={pickPlace}
          searchNote={searchNote}
          years={years}
          onYears={setYears}
          health={health}
          warnings={task?.warnings}
        />

        {section === 'map' ? (
          <div className="canvas">
            <MapWorkspace
              {...shared}
              parcels={parcels}
              saved={savedFields}
              geometry={geometry}
              selectedLabel={selectedLabel}
              regionOutline={regionOutline}
              regionKey={regionKey}
              regionTitle={region}
              onPickRegion={pickRegion}
              onBack={() => setSection('overview')}
              flyTo={flyTo}
              onPickParcel={pickParcel}
              onOpenSaved={(id) => openField({ id })}
              onDrawn={onDrawn}
              onDiscover={discover}
              discovering={discovering}
              discoverNote={discoverNote}
            />
          </div>
        ) : (
          <div className="canvas scroll">
            {section === 'overview' && (
              <Overview summary={summary} onGoMap={() => setSection('map')} />
            )}
            {section === 'fields' && (
              <Fields
                summary={summary}
                selectedId={selected?.id}
                onOpen={openField}
                onRename={renameField}
                onDelete={deleteField}
                onAnalyze={analyzeSaved}
                onGoMap={() => setSection('map')}
              />
            )}
            {section === 'analytics' && (
              <Analytics summary={summary} onOpenField={openField} />
            )}
            {section === 'reports' && (
              <Reports summary={summary} onGoMap={() => setSection('map')} />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/** Рамка контура — нужна, чтобы карта перелетела к выбранному участку. */
function bboxOf(geometry) {
  const rings = geometry.type === 'Polygon' ? geometry.coordinates : geometry.coordinates.flat()
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
