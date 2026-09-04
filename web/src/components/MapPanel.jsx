import { useEffect, useRef, useState } from 'react'
// В maplibre-gl 6 умолчательного экспорта больше нет — только именованные.
import { Map as MapLibreMap, NavigationControl, ScaleControl } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

// Подложки. Обе проверены на доступность до начала работы: OSM для ориентировки
// по дорогам и посёлкам, спутниковая — чтобы глазами убедиться, что выбранный
// контур действительно поле, а не лесополоса.
const BASEMAPS = {
  osm: {
    title: 'Карта',
    tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
    attribution: '© OpenStreetMap',
  },
  satellite: {
    title: 'Снимок',
    tiles: [
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    ],
    attribution: '© Esri World Imagery',
  },
}

function styleFor(kind) {
  const base = BASEMAPS[kind]
  return {
    version: 8,
    sources: {
      base: { type: 'raster', tiles: base.tiles, tileSize: 256, attribution: base.attribution },
    },
    layers: [{ id: 'base', type: 'raster', source: 'base' }],
  }
}

const EMPTY = { type: 'FeatureCollection', features: [] }

/**
 * Карта: поиск региона приводит сюда, отсюда же выбирается или рисуется контур.
 *
 * Оба сценария постановки живут на этой карте и дальше сходятся в один путь:
 * выбранный контур из OSM и нарисованный вручную полигон уходят на анализ
 * одинаково.
 */
export default function MapPanel({
  parcels,
  selectedGeometry,
  regionOutline,
  flyTo,
  onPickParcel,
  onDrawn,
  onDiscover,
  discovering,
  discoverNote,
}) {
  const container = useRef(null)
  const map = useRef(null)
  const [ready, setReady] = useState(false)
  const [basemap, setBasemap] = useState('osm')
  const [drawing, setDrawing] = useState(false)
  // Точки рисуемого контура держим в ref, а не только в состоянии: обработчик
  // клика подписан на карту один раз и в замыкании видел бы устаревший массив.
  const draftPoints = useRef([])
  const [draftCount, setDraftCount] = useState(0)
  const drawingRef = useRef(false)

  // ------------------------------------------------------------ создание карты
  useEffect(() => {
    const instance = new MapLibreMap({
      container: container.current,
      style: styleFor('osm'),
      // Центр по умолчанию — юг России, но это только стартовый вид: ни одного
      // региона в логике сервиса не зашито, работать он обязан везде.
      center: [39.7, 47.2],
      zoom: 8,
      attributionControl: { compact: true },
    })
    instance.addControl(new NavigationControl({ showCompass: false }), 'top-right')
    instance.addControl(new ScaleControl({ unit: 'metric' }), 'bottom-right')

    instance.on('load', () => {
      addLayers(instance)
      setReady(true)
    })
    map.current = instance
    return () => instance.remove()
  }, [])

  // Смена подложки пересоздаёт стиль, поэтому слои данных нужно вернуть обратно.
  useEffect(() => {
    if (!map.current || !ready) return
    map.current.setStyle(styleFor(basemap))
    map.current.once('styledata', () => addLayers(map.current))
  }, [basemap, ready])

  function addLayers(instance) {
    const add = (id, data) => {
      if (!instance.getSource(id)) instance.addSource(id, { type: 'geojson', data })
    }
    add('region', EMPTY)
    add('parcels', EMPTY)
    add('selected', EMPTY)
    add('draft', EMPTY)

    if (!instance.getLayer('region-line')) {
      instance.addLayer({
        id: 'region-line',
        type: 'line',
        source: 'region',
        paint: { 'line-color': '#3f7d4e', 'line-width': 1.5, 'line-dasharray': [3, 2] },
      })
    }
    if (!instance.getLayer('parcels-fill')) {
      instance.addLayer({
        id: 'parcels-fill',
        type: 'fill',
        source: 'parcels',
        paint: { 'fill-color': '#3f7d4e', 'fill-opacity': 0.18 },
      })
      instance.addLayer({
        id: 'parcels-line',
        type: 'line',
        source: 'parcels',
        paint: { 'line-color': '#3f7d4e', 'line-width': 1.2 },
      })
    }
    if (!instance.getLayer('selected-fill')) {
      instance.addLayer({
        id: 'selected-fill',
        type: 'fill',
        source: 'selected',
        paint: { 'fill-color': '#d98324', 'fill-opacity': 0.28 },
      })
      instance.addLayer({
        id: 'selected-line',
        type: 'line',
        source: 'selected',
        paint: { 'line-color': '#b8651a', 'line-width': 2.4 },
      })
    }
    if (!instance.getLayer('draft-line')) {
      instance.addLayer({
        id: 'draft-fill',
        type: 'fill',
        source: 'draft',
        paint: { 'fill-color': '#2f6fb3', 'fill-opacity': 0.2 },
      })
      instance.addLayer({
        id: 'draft-line',
        type: 'line',
        source: 'draft',
        paint: { 'line-color': '#2f6fb3', 'line-width': 2, 'line-dasharray': [2, 1.5] },
      })
      instance.addLayer({
        id: 'draft-points',
        type: 'circle',
        source: 'draft',
        filter: ['==', '$type', 'Point'],
        paint: { 'circle-radius': 4.5, 'circle-color': '#2f6fb3', 'circle-stroke-width': 2, 'circle-stroke-color': '#fff' },
      })
    }
  }

  const setData = (id, data) => {
    const source = map.current?.getSource(id)
    if (source) source.setData(data)
  }

  // ------------------------------------------------------------ данные слоёв
  useEffect(() => {
    if (!ready) return
    setData('parcels', {
      type: 'FeatureCollection',
      features: (parcels || []).map((p) => ({
        type: 'Feature',
        id: p.id,
        properties: { id: p.id, area_ha: p.area_ha, crop_hint: p.crop_hint, name: p.name },
        geometry: p.geometry,
      })),
    })
  }, [parcels, ready])

  useEffect(() => {
    if (!ready) return
    setData(
      'selected',
      selectedGeometry
        ? { type: 'Feature', properties: {}, geometry: selectedGeometry }
        : EMPTY,
    )
  }, [selectedGeometry, ready])

  useEffect(() => {
    if (!ready) return
    setData('region', regionOutline ? { type: 'Feature', properties: {}, geometry: regionOutline } : EMPTY)
  }, [regionOutline, ready])

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
      if (hits.length) onPickParcel?.(hits[0].properties.id)
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
  }, [ready, onPickParcel])

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

  return (
    <div className="map-wrap">
      <div ref={container} className="map" />

      <div className="map-tools">
        {!drawing ? (
          <>
            <button className="primary" onClick={discover} disabled={discovering}>
              {discovering ? 'Ищу поля…' : 'Найти поля в этом районе'}
            </button>
            <button onClick={startDrawing}>Нарисовать полигон</button>
          </>
        ) : (
          <>
            <button className="primary" onClick={finishDrawing} disabled={draftCount < 3}>
              Завершить контур
            </button>
            <button onClick={undoPoint} disabled={!draftCount}>Убрать точку</button>
            <button onClick={cancelDrawing}>Отмена</button>
          </>
        )}
        <button onClick={() => setBasemap(basemap === 'osm' ? 'satellite' : 'osm')}>
          {BASEMAPS[basemap === 'osm' ? 'satellite' : 'osm'].title}
        </button>
      </div>

      {drawing && (
        <div className="map-hint">
          Кликайте по карте, обводя поле. Поставлено точек: {draftCount}. Двойной клик или
          «Завершить контур» замыкает полигон.
        </div>
      )}
      {!drawing && discoverNote && <div className="map-hint warn">{discoverNote}</div>}
      {!drawing && !discoverNote && parcels?.length > 0 && (
        <div className="map-hint">
          Найдено контуров: {parcels.length}. Кликните по любому, чтобы выбрать поле.
        </div>
      )}
    </div>
  )
}
