import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from 'react'

import { api, pollTask } from './api.js'
import { REGION_PRESETS, fieldState } from './dict.js'
import Fields from './components/Fields.jsx'
import Overview from './components/Overview.jsx'
import Reports from './components/Reports.jsx'
import Sidebar from './components/Sidebar.jsx'
import Topbar from './components/Topbar.jsx'

/*
 * Тяжёлые разделы грузятся отдельными файлами.
 *
 * MapLibre (карта) и Recharts (графики) вместе занимают около 85 % бандла:
 * одним куском он весил 1,66 МБ, и всё это скачивалось до первой отрисовки —
 * даже если человек открыл «Обзор» и на карту в этот заход вообще не пойдёт.
 * `lazy` разносит их по отдельным файлам, которые браузер запрашивает только
 * при входе в соответствующий раздел.
 *
 * Делить именно по этим двум точкам, а не по всем пяти разделам: «Обзор»,
 * «Участки» и «Отчёты» не тянут ни одной тяжёлой библиотеки, отдельные файлы
 * для них дали бы лишние запросы без выигрыша по весу.
 */
const MapWorkspace = lazy(() => import('./components/MapWorkspace.jsx'))
const Analytics = lazy(() => import('./components/Analytics.jsx'))

const VERSION = '1.0'

// Заголовок и подзаголовок каждого раздела. Держим в одном месте: они попадают
// и в шапку, и в логику раздела, и расхождение между ними сразу заметно.
const TITLES = {
  overview: ['Добро пожаловать', 'Независимые данные и прогнозы для вашего агробизнеса'],
  map: ['Карта', 'Найдите готовый контур или нарисуйте свой — дальше сервис всё сделает сам'],
  fields: ['Участки', 'Сохранённые поля: пересчёт, переименование, удаление'],
  analytics: ['Аналитика', 'Показатели выбранного поля по сезонам и сравнение полей'],
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
  const [rechecking, setRechecking] = useState(false)

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

  // Догрузка кода карты и графиков, когда браузер освободился.
  //
  // Разделение кода само по себе переносит ожидание со старта на первый вход в
  // раздел. Чтобы не менять шило на мыло, оба файла запрашиваются в простое
  // сразу после первой отрисовки: стартовый экран уже нарисован и ничего не
  // ждёт, а к моменту клика по «Карте» файл обычно уже лежит в кэше и
  // заставка не успевает мелькнуть. requestIdleCallback есть не везде
  // (Safari до 17), поэтому запасной вариант — обычный таймер.
  useEffect(() => {
    const warm = () => {
      import('./components/MapWorkspace.jsx')
      import('./components/Analytics.jsx')
    }
    if (typeof requestIdleCallback === 'function') {
      const id = requestIdleCallback(warm, { timeout: 3000 })
      return () => cancelIdleCallback(id)
    }
    const id = setTimeout(warm, 1200)
    return () => clearTimeout(id)
  }, [])

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

  async function recheckHealth() {
    setRechecking(true)
    try {
      setHealth(await api.health(true))
    } catch {
      // Молчим намеренно: неудачная перепроверка не должна ронять экран,
      // а прошлое состояние остаётся на месте и подписано своим временем.
    } finally {
      setRechecking(false)
    }
  }

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
        rechecking={rechecking}
        onRecheck={recheckHealth}
      />

      <div className="main">
        {/* showRegion: поиск региона по названию остаётся только на карте.
            На «Обзоре» его место занял выбор в блоке быстрого старта, а в
            списках и отчётах он ничего не делает — только уводит на карту.
            У самой карты есть свой список регионов ЮФО, но по нему не попасть
            в конкретную станицу, поэтому поиск здесь и нужен. */}
        <Topbar
          title={title}
          subtitle={subtitle}
          showRegion={section === 'map'}
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
            <Suspense fallback={<Loading what="карту" />}>
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
            </Suspense>
          </div>
        ) : (
          <div className="canvas scroll">
            {section === 'overview' && (
              <Overview
                summary={summary}
                onGoMap={() => setSection('map')}
                onOpenField={openField}
                region={region}
                onSearchRegion={searchRegion}
                places={places}
                searching={searching}
                searchNote={searchNote}
                onPickPlace={pickPlace}
              />
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
              <Suspense fallback={<Loading what="графики" />}>
                <Analytics summary={summary} onOpenField={openField} />
              </Suspense>
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

/**
 * Заставка на время догрузки кода раздела.
 *
 * Показывается вместо пустоты, пока браузер качает отдельный файл карты или
 * графиков. Обычно она не успевает появиться — код уже догружен в простое, —
 * но на медленной связи и при первом заходе именно она объясняет паузу, а не
 * оставляет человека перед пустым прямоугольником.
 */
function Loading({ what }) {
  return (
    <div className="lazy-hold">
      <div className="lazy-dots" aria-hidden="true"><i /><i /><i /></div>
      <p>Загружаю {what}…</p>
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
