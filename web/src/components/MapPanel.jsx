import { useEffect, useMemo, useRef, useState } from 'react'
// В maplibre-gl 6 умолчательного экспорта больше нет — только именованные.
import { AttributionControl, Map as MapLibreMap, Popup, setWorkerUrl } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

import { FIELD_STATE, FIELD_STATE_ORDER, REGION_PRESETS, cropTitle } from '../dict.js'
import { useTheme } from '../theme.js'
import {
  IconBack,
  IconCheck,
  IconClose,
  IconLayers,
  IconMinus,
  IconPlus,
  IconPolygon,
  IconTarget,
  IconUndo,
} from './icons.jsx'

// Воркер maplibre — отдельный файл, и его адрес библиотека вычисляет как
// соседний с собственным модулем. После сборки библиотека лежит внутри общего
// бандла, соседа рядом нет, и запрос уходит в 404.
//
// Ошибка тихая и обманчивая: растровая подложка грузится в главном потоке и
// рисуется как ни в чём не бывало, а вот все слои GeoJSON — найденные контуры
// полей, выбранное поле, рисуемый полигон — разбираются в воркере и просто не
// появляются. Карта при этом выглядит рабочей.
//
// Оба файла воркера кладёт рядом плагин сборки (см. vite.config.js), здесь
// остаётся только назвать библиотеке адрес.
setWorkerUrl(`${import.meta.env.BASE_URL}maplibre/maplibre-gl-worker.mjs`)

// Подложки. Обе проверены на доступность с рабочей машины: спутниковая — чтобы
// глазами убедиться, что выбранный контур действительно поле, а не лесополоса;
// схема — запасной вариант, она в разы легче и выручает на слабом канале.
//
// По умолчанию включён снимок: на нём поля читаются как поля, и именно так
// выглядит экран в макете. Переключатель живёт в плашке слева.
const BASEMAPS = {
  satellite: {
    title: 'Снимок',
    tiles: [
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    ],
    attribution: '© Esri, Maxar, Earthstar Geographics',
    maxzoom: 19,
  },
  osm: {
    title: 'Схема',
    tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
    attribution: '© OpenStreetMap',
    maxzoom: 19,
  },
}

/**
 * Стиль карты для выбранной подложки.
 *
 * Снимок от темы не зависит: поле на нём выглядит полем при любом оформлении,
 * и перекрашивать реальность было бы враньём. А вот схема OpenStreetMap
 * нарисована для светлого экрана — её белая заливка в тёмной теме светится
 * ярче всего остального интерфейса вместе взятого. Своих цветов у чужих
 * растровых тайлов нет, поэтому подложка приглушается средствами самой карты:
 * потолок яркости, чуть меньше насыщенности и контраста.
 */
function styleFor(kind, theme = 'light') {
  const base = BASEMAPS[kind]
  const dim = kind === 'osm' && theme === 'dark'
  return {
    version: 8,
    sources: {
      base: {
        type: 'raster',
        tiles: base.tiles,
        tileSize: 256,
        maxzoom: base.maxzoom,
        attribution: base.attribution,
      },
    },
    layers: [
      {
        id: 'base',
        type: 'raster',
        source: 'base',
        paint: dim
          ? { 'raster-brightness-max': 0.68, 'raster-saturation': -0.2, 'raster-contrast': -0.08 }
          : {},
      },
    ],
  }
}

const EMPTY = { type: 'FeatureCollection', features: [] }

// Цвет заливки и контура берётся из одного словаря состояний — того же, по
// которому подписана легенда в углу. Разъехаться они физически не могут.
const colorMatch = (key) => {
  const expr = ['match', ['get', 'state']]
  for (const state of FIELD_STATE_ORDER) {
    if (state === 'nodata') continue
    expr.push(state, FIELD_STATE[state][key])
  }
  expr.push(FIELD_STATE.nodata[key]) // «нет данных» — умолчание
  return expr
}

// Ступени масштабной линейки: 1-2-5 в каждом порядке. Отрезок подписывается
// круглым числом, иначе линейка показывает «1,37 км» и перестаёт быть линейкой.
const SCALE_STEPS = [
  1, 2, 5, 10, 20, 50, 100, 200, 500,
  1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000,
]
const SCALE_MAX_PX = 120

/** Подпись и длина отрезка линейки для текущего масштаба карты. */
function scaleFor(instance) {
  const height = instance.getContainer().clientHeight
  if (!height) return null
  // Меряем по середине экрана: у веб-меркатора масштаб зависит от широты, и
  // линейка, посчитанная по верхнему краю, врала бы тем сильнее, чем севернее.
  const y = height / 2
  const meters = instance.unproject([0, y]).distanceTo(instance.unproject([SCALE_MAX_PX, y]))
  if (!Number.isFinite(meters) || meters <= 0) return null
  let step = SCALE_STEPS[0]
  for (const candidate of SCALE_STEPS) if (candidate <= meters) step = candidate
  return {
    width: Math.round((step / meters) * SCALE_MAX_PX),
    label: step >= 1000 ? `${step / 1000} км` : `${step} м`,
  }
}

/** Все вершины полигона списком — по ним рисуются кружки на выбранном контуре. */
function verticesOf(geometry) {
  if (!geometry) return []
  const rings = geometry.type === 'Polygon' ? geometry.coordinates : geometry.coordinates.flat()
  const points = []
  for (const ring of rings) {
    // Последняя точка кольца повторяет первую — рисовать её второй раз незачем.
    for (const point of ring.slice(0, -1)) points.push(point)
  }
  return points
}

/** Самая северная вершина: над ней вешается карточка поля, как в макете. */
function topPointOf(geometry) {
  const points = verticesOf(geometry)
  if (!points.length) return null
  return points.reduce((best, point) => (point[1] > best[1] ? point : best), points[0])
}

/**
 * Карта: поиск региона приводит сюда, отсюда же выбирается или рисуется контур.
 *
 * Оба сценария постановки живут на этой карте и дальше сходятся в один путь:
 * выбранный контур из OSM и нарисованный вручную полигон уходят на анализ
 * одинаково.
 */
export default function MapPanel({
  parcels,
  saved,
  selectedGeometry,
  selectedLabel,
  regionOutline,
  regionKey,
  regionTitle,
  onPickRegion,
  onBack,
  flyTo,
  onPickParcel,
  onOpenSaved,
  onDrawn,
  onDiscover,
  discovering,
  discoverNote,
}) {
  const container = useRef(null)
  const map = useRef(null)
  const popup = useRef(null)
  const [ready, setReady] = useState(false)
  const [basemap, setBasemap] = useState('satellite')
  const { theme } = useTheme()
  // Тема важна только схеме: на снимке менять нечего, и пересобирать из-за
  // переключения темы стиль карты со спутником значило бы моргать подложкой
  // на ровном месте.
  const basemapTone = basemap === 'osm' ? theme : 'light'
  const [scale, setScale] = useState(null)
  const [drawing, setDrawing] = useState(false)
  // Точки рисуемого контура держим в ref, а не только в состоянии: обработчик
  // клика подписан на карту один раз и в замыкании видел бы устаревший массив.
  const draftPoints = useRef([])
  const [draftCount, setDraftCount] = useState(0)
  const drawingRef = useRef(false)

  // Последнее, что положили в каждый слой. Это и есть источник правды: слои
  // карты могут быть пересозданы в любой момент, состояние React — нет.
  const layerData = useRef({
    region: EMPTY,
    parcels: EMPTY,
    selected: EMPTY,
    vertices: EMPTY,
    draft: EMPTY,
  })

  // ------------------------------------------------------------ создание карты
  useEffect(() => {
    const instance = new MapLibreMap({
      container: container.current,
      style: styleFor('satellite'),
      // Центр по умолчанию — юг России, но это только стартовый вид: ни одного
      // региона в логике сервиса не зашито, работать он обязан везде.
      center: [39.7, 47.2],
      zoom: 8,
      // Штатные кнопки зума убраны: в макете зум живёт в вертикальной плашке
      // слева вместе с рисованием, а два набора кнопок на одной карте — это
      // вопрос «а эти чем отличаются» на защите.
      attributionControl: false,
    })
    // Атрибуция обязательна по условиям обоих источников тайлов. Компактная и
    // в левом нижнем углу — там же, где линейка, чтобы не спорить с легендой.
    instance.addControl(new AttributionControl({ compact: true }), 'bottom-left')

    // Линейка своя, а не штатная: у maplibre подпись зашита по-английски
    // («2 km»), а весь остальной экран русский. Считать метры на пиксель всё
    // равно приходится одной строкой, так что это дешевле любого обходного пути.
    const refreshScale = () => setScale(scaleFor(instance))
    instance.on('move', refreshScale)
    instance.on('resize', refreshScale)

    instance.on('load', () => {
      addLayers(instance)
      refreshScale()
      setReady(true)
    })
    map.current = instance
    return () => {
      popup.current?.remove()
      instance.remove()
    }
  }, [])

  // Смена подложки пересоздаёт стиль, поэтому слои данных нужно вернуть обратно.
  useEffect(() => {
    if (!map.current || !ready) return
    map.current.setStyle(styleFor(basemap, basemapTone))
    map.current.once('styledata', () => addLayers(map.current))
  }, [basemap, basemapTone, ready])

  function addLayers(instance) {
    // Источники создаются сразу с актуальными данными, а не пустыми.
    //
    // Пустыми было нельзя по двум причинам, и обе выглядят как «карта потеряла
    // полигон». Первая: слои создаются в обработчике load, то есть позже, чем
    // отработали эффекты с данными, и выбранный участок, пришедший до загрузки
    // карты (например, при открытии поля из раздела «Участки»), просто не
    // доезжал. Вторая: смена подложки пересоздаёт стиль вместе со всеми
    // источниками, а эффекты при этом не перезапускаются — зависимости не
    // менялись, — и переключение «Снимок / Схема» стирало с карты и найденные
    // контуры, и выбранное поле.
    const add = (id) => {
      if (!instance.getSource(id)) {
        instance.addSource(id, { type: 'geojson', data: layerData.current[id] })
      }
    }
    add('region')
    add('parcels')
    add('selected')
    add('vertices')
    add('draft')

    if (!instance.getLayer('region-line')) {
      instance.addLayer({
        id: 'region-line',
        type: 'line',
        source: 'region',
        paint: {
          'line-color': '#ffffff',
          'line-width': 1.4,
          'line-opacity': 0.7,
          'line-dasharray': [3, 2],
        },
      })
    }
    // Поля: цвет заливки и контура — по состоянию, как в легенде.
    if (!instance.getLayer('parcels-fill')) {
      instance.addLayer({
        id: 'parcels-fill',
        type: 'fill',
        source: 'parcels',
        paint: { 'fill-color': colorMatch('fill') },
      })
      instance.addLayer({
        id: 'parcels-line',
        type: 'line',
        source: 'parcels',
        paint: { 'line-color': colorMatch('color'), 'line-width': 1.6 },
      })
    }
    // Выбранное поле — сине-фиолетовое, контур ярче, по вершинам кружки.
    if (!instance.getLayer('selected-fill')) {
      instance.addLayer({
        id: 'selected-fill',
        type: 'fill',
        source: 'selected',
        paint: { 'fill-color': '#4f46e5', 'fill-opacity': 0.42 },
      })
      instance.addLayer({
        id: 'selected-line',
        type: 'line',
        source: 'selected',
        paint: { 'line-color': '#8b8bf5', 'line-width': 2.2 },
      })
      instance.addLayer({
        id: 'selected-points',
        type: 'circle',
        source: 'vertices',
        paint: {
          'circle-radius': 3.4,
          'circle-color': '#ffffff',
          'circle-stroke-width': 1.6,
          'circle-stroke-color': '#6366f1',
        },
      })
    }
    if (!instance.getLayer('draft-line')) {
      instance.addLayer({
        id: 'draft-fill',
        type: 'fill',
        source: 'draft',
        paint: { 'fill-color': '#4f46e5', 'fill-opacity': 0.3 },
      })
      instance.addLayer({
        id: 'draft-line',
        type: 'line',
        source: 'draft',
        paint: { 'line-color': '#a5b4fc', 'line-width': 2, 'line-dasharray': [2, 1.5] },
      })
      instance.addLayer({
        id: 'draft-points',
        type: 'circle',
        source: 'draft',
        filter: ['==', '$type', 'Point'],
        paint: {
          'circle-radius': 4.5,
          'circle-color': '#4f46e5',
          'circle-stroke-width': 2,
          'circle-stroke-color': '#fff',
        },
      })
    }
  }

  const setData = (id, data) => {
    layerData.current[id] = data
    const source = map.current?.getSource(id)
    if (source) source.setData(data)
  }

  // ------------------------------------------------------------ данные слоёв
  //
  // Сохранённые участки и найденные контуры лежат в одном слое: рисуются они
  // одинаково, различает их только состояние (у сохранённых оно посчитано, у
  // найденных — ещё нет) и то, что происходит по клику. Два слоя с одинаковым
  // оформлением означали бы два обработчика клика и два места, где можно
  // разойтись в цвете.
  useEffect(() => {
    const features = []
    for (const field of saved || []) {
      features.push({
        type: 'Feature',
        id: `saved-${field.id}`,
        properties: { kind: 'saved', pid: field.id, state: field.state || 'nodata' },
        geometry: field.geometry,
      })
    }
    for (const parcel of parcels || []) {
      features.push({
        type: 'Feature',
        id: `osm-${parcel.id}`,
        // Найденный контур ещё никто не считал — честное «нет данных», серый.
        properties: { kind: 'osm', pid: parcel.id, state: 'nodata' },
        geometry: parcel.geometry,
      })
    }
    setData('parcels', { type: 'FeatureCollection', features })
  }, [parcels, saved])

  useEffect(() => {
    setData(
      'selected',
      selectedGeometry ? { type: 'Feature', properties: {}, geometry: selectedGeometry } : EMPTY,
    )
    setData('vertices', {
      type: 'FeatureCollection',
      features: verticesOf(selectedGeometry).map((coordinates) => ({
        type: 'Feature',
        properties: {},
        geometry: { type: 'Point', coordinates },
      })),
    })
  }, [selectedGeometry])

  useEffect(() => {
    setData('region', regionOutline ? { type: 'Feature', properties: {}, geometry: regionOutline } : EMPTY)
  }, [regionOutline])

  // ------------------------------------------- карточка над выбранным полем
  //
  // Это штатный popup maplibre, а не свой div поверх карты: он сам держится за
  // точку на местности при перетаскивании и зуме, и у него есть «хвостик».
  // Свой пришлось бы пересчитывать на каждом кадре движения карты.
  useEffect(() => {
    if (!ready) return
    const anchorPoint = topPointOf(selectedGeometry)
    if (!anchorPoint || !selectedLabel?.name) {
      popup.current?.remove()
      popup.current = null
      return
    }
    const node = document.createElement('div')
    node.className = 'field-tip'
    const title = document.createElement('strong')
    title.textContent = selectedLabel.name
    const sub = document.createElement('span')
    // Вторая строка есть всегда: у контура из OSM культура бывает не размечена,
    // и честное «культура не указана» лучше, чем схлопнувшаяся карточка.
    sub.textContent = cropTitle(selectedLabel.crop) || 'культура не указана'
    node.append(title, sub)

    if (!popup.current) {
      popup.current = new Popup({
        closeButton: false,
        closeOnClick: false,
        // Сторону maplibre выбирает сам: у верхней кромки карточка, жёстко
        // прибитая сверху к контуру, уезжала за край и обрезалась.
        offset: 14,
        maxWidth: '250px',
        className: 'field-popup',
      }).addTo(map.current)
    }
    popup.current.setLngLat(anchorPoint).setDOMContent(node)
  }, [ready, selectedGeometry, selectedLabel])

  // Перелёт к найденному региону или к выбранному полю.
  useEffect(() => {
    if (!ready || !flyTo) return
    const [west, south, east, north] = flyTo.bbox
    map.current.fitBounds([[west, south], [east, north]], { padding: 48, duration: 900, maxZoom: 15 })
  }, [flyTo, ready])

  // ----------------------------------------------------------------- выбор поля
  useEffect(() => {
    if (!ready) return
    const instance = map.current

    const click = (event) => {
      if (drawingRef.current) return
      const hits = instance.queryRenderedFeatures(event.point, { layers: ['parcels-fill'] })
      if (!hits.length) return
      const { kind, pid } = hits[0].properties
      // Сохранённый участок открывается вместе с прошлым разбором, найденный —
      // становится черновиком под анализ. Оба пути дальше одинаковы.
      if (kind === 'saved') onOpenSaved?.(pid)
      else onPickParcel?.(pid)
    }
    const enter = () => { if (!drawingRef.current) instance.getCanvas().style.cursor = 'pointer' }
    const leave = () => { if (!drawingRef.current) instance.getCanvas().style.cursor = '' }

    instance.on('click', 'parcels-fill', click)
    instance.on('mouseenter', 'parcels-fill', enter)
    instance.on('mouseleave', 'parcels-fill', leave)
    return () => {
      instance.off('click', 'parcels-fill', click)
      instance.off('mouseenter', 'parcels-fill', enter)
      instance.off('mouseleave', 'parcels-fill', leave)
    }
  }, [ready, onPickParcel, onOpenSaved])

  // ------------------------------------------------------------- рисование
  //
  // Рисуем сами, без плагина. Готовые библиотеки рисования тянут за собой
  // редактирование вершин, дырки и мультиполигоны — всё то, чего в сценарии нет,
  // зато есть их несовместимость с мажорными версиями maplibre. Здесь нужно
  // ровно одно: замкнуть контур кликами.
  useEffect(() => {
    if (!ready) return
    const instance = map.current

    const onClick = (event) => {
      if (!drawingRef.current) return
      draftPoints.current = [...draftPoints.current, [event.lngLat.lng, event.lngLat.lat]]
      setDraftCount(draftPoints.current.length)
      renderDraft()
    }
    const onDouble = (event) => {
      if (!drawingRef.current) return
      event.preventDefault()
      finishDrawing()
    }
    instance.on('click', onClick)
    instance.on('dblclick', onDouble)
    return () => {
      instance.off('click', onClick)
      instance.off('dblclick', onDouble)
    }
  }, [ready])

  function renderDraft() {
    const points = draftPoints.current
    const features = points.map((coordinates) => ({
      type: 'Feature',
      properties: {},
      geometry: { type: 'Point', coordinates },
    }))
    if (points.length >= 3) {
      features.push({
        type: 'Feature',
        properties: {},
        geometry: { type: 'Polygon', coordinates: [[...points, points[0]]] },
      })
    } else if (points.length === 2) {
      features.push({
        type: 'Feature',
        properties: {},
        geometry: { type: 'LineString', coordinates: points },
      })
    }
    setData('draft', { type: 'FeatureCollection', features })
  }

  function startDrawing() {
    draftPoints.current = []
    setDraftCount(0)
    renderDraft()
    drawingRef.current = true
    setDrawing(true)
    // Двойной клик по умолчанию приближает карту — при рисовании это мешает
    // завершить контур.
    map.current.doubleClickZoom.disable()
    map.current.getCanvas().style.cursor = 'crosshair'
  }

  function stopDrawing() {
    drawingRef.current = false
    setDrawing(false)
    map.current.doubleClickZoom.enable()
    map.current.getCanvas().style.cursor = ''
  }

  function finishDrawing() {
    const points = draftPoints.current
    if (points.length < 3) return
    onDrawn?.({ type: 'Polygon', coordinates: [[...points, points[0]]] })
    draftPoints.current = []
    setDraftCount(0)
    setData('draft', EMPTY)
    stopDrawing()
  }

  function cancelDrawing() {
    draftPoints.current = []
    setDraftCount(0)
    setData('draft', EMPTY)
    stopDrawing()
  }

  function undoPoint() {
    draftPoints.current = draftPoints.current.slice(0, -1)
    setDraftCount(draftPoints.current.length)
    renderDraft()
  }

  function discover() {
    const bounds = map.current.getBounds()
    onDiscover?.([bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()])
  }

  // Подсказка внизу экрана всего одна и в один момент времени: что именно
  // показать, решаем здесь, а не тремя условиями в разметке.
  const hint = useMemo(() => {
    if (drawing) {
      return {
        tone: '',
        text: `Кликайте по карте, обводя поле. Поставлено точек: ${draftCount}. Двойной клик или «Завершить контур» замыкает полигон.`,
      }
    }
    if (discovering) return { tone: '', text: 'Ищу поля в этой рамке карты…' }
    if (discoverNote) return { tone: 'warn', text: discoverNote }
    // Как только контур выбран, подсказка «кликните по любому» своё отработала
    // и только мешает: она висит там же, где карточка выбранного поля.
    if (parcels?.length && !selectedGeometry) {
      return { tone: '', text: `Найдено контуров: ${parcels.length}. Кликните по любому, чтобы выбрать поле.` }
    }
    return null
  }, [drawing, draftCount, discovering, discoverNote, parcels, selectedGeometry])

  return (
    <div className="map-wrap">
      {/* Верхняя панель по макету: только «назад» и регион, больше ничего. */}
      <div className="map-topbar">
        <button className="map-back" onClick={onBack} title="К обзору" aria-label="Назад">
          <IconBack />
        </button>
        <label className="map-region">
          <select
            value={regionKey || ''}
            onChange={(event) => onPickRegion?.(event.target.value)}
          >
            {/* Регион мог приехать из поиска в шапке — тогда пресетом он не
                является, и подменять его название на чужое нельзя. */}
            {!regionKey && <option value="">{regionTitle || 'Выберите регион'}</option>}
            {REGION_PRESETS.map((preset) => (
              <option key={preset.key} value={preset.key}>
                {preset.title}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="map-stage">
        <div ref={container} className="map" />

        {/* Вертикальная плашка слева: поиск полей, рисование, зум, подложка. */}
        <div className="map-rail">
          {!drawing ? (
            <>
              <button
                className="rail-btn"
                onClick={discover}
                disabled={discovering}
                data-label="Найти поля в этом районе"
              >
                <IconTarget />
              </button>
              <button className="rail-btn" onClick={startDrawing} data-label="Нарисовать свой контур">
                <IconPolygon />
              </button>
            </>
          ) : (
            <>
              <button
                className="rail-btn accent"
                onClick={finishDrawing}
                disabled={draftCount < 3}
                data-label="Завершить контур"
              >
                <IconCheck />
              </button>
              <button
                className="rail-btn"
                onClick={undoPoint}
                disabled={!draftCount}
                data-label="Убрать последнюю точку"
              >
                <IconUndo />
              </button>
              <button className="rail-btn" onClick={cancelDrawing} data-label="Отменить рисование">
                <IconClose />
              </button>
            </>
          )}

          <span className="rail-sep" />

          <button
            className="rail-btn"
            onClick={() => map.current?.zoomIn()}
            data-label="Приблизить"
          >
            <IconPlus />
          </button>
          <button
            className="rail-btn"
            onClick={() => map.current?.zoomOut()}
            data-label="Отдалить"
          >
            <IconMinus />
          </button>

          <span className="rail-sep" />

          <button
            className="rail-btn"
            onClick={() => setBasemap(basemap === 'osm' ? 'satellite' : 'osm')}
            data-label={`Подложка: ${BASEMAPS[basemap === 'osm' ? 'satellite' : 'osm'].title}`}
          >
            <IconLayers />
          </button>
        </div>

        {/* Легенда — то же соответствие цвета и состояния, что на самих полях. */}
        <div className="map-legend">
          {FIELD_STATE_ORDER.map((key) => (
            <div className="legend-row" key={key}>
              <span className="swatch" style={{ background: FIELD_STATE[key].color }} />
              {FIELD_STATE[key].label}
            </div>
          ))}
        </div>

        {/* Масштабная линейка: подпись над отрезком, как в макете. */}
        {scale && (
          <div className="map-scale">
            <span>{scale.label}</span>
            <i style={{ width: `${scale.width}px` }} />
          </div>
        )}

        {hint && <div className={`map-hint ${hint.tone}`}>{hint.text}</div>}
      </div>
    </div>
  )
}
