import AnomalyFeed from './AnomalyFeed.jsx'
import FieldPanel from './FieldPanel.jsx'
import MapPanel from './MapPanel.jsx'
import SeriesChart from './SeriesChart.jsx'
import WeatherPanel from './WeatherPanel.jsx'

/**
 * Раздел «Карта» — основной сценарий целиком на одном экране.
 *
 * Карта сверху, разбор выбранного поля снизу. Разносить их по разным разделам
 * нельзя: пользователь выбирает контур и тут же должен увидеть, что получилось,
 * а не идти за результатом в другое меню.
 */
export default function MapWorkspace(props) {
  const {
    parcels,
    saved,
    geometry,
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
    result,
    activeAnomaly,
    setActiveAnomaly,
  } = props

  return (
    <div className="map-screen">
      <MapPanel
        parcels={parcels}
        saved={saved}
        selectedGeometry={geometry}
        selectedLabel={selectedLabel}
        regionOutline={regionOutline}
        regionKey={regionKey}
        regionTitle={regionTitle}
        onPickRegion={onPickRegion}
        onBack={onBack}
        flyTo={flyTo}
        onPickParcel={onPickParcel}
        onOpenSaved={onOpenSaved}
        onDrawn={onDrawn}
        onDiscover={onDiscover}
        discovering={discovering}
        discoverNote={discoverNote}
      />

      <div className="map-bottom">
        <div className="map-charts">
          <FieldPanel {...props} />
          {result && (
            <>
              <SeriesChart
                series={result.series}
                anomalies={result.anomalies || []}
                activeAnomaly={activeAnomaly}
                onPickAnomaly={setActiveAnomaly}
              />
              <WeatherPanel
                weather={result.weather}
                anomalies={result.anomalies || []}
                activeAnomaly={activeAnomaly}
              />
            </>
          )}
        </div>

        <div className="map-feed">
          {result ? (
            <AnomalyFeed
              anomalies={result.anomalies || []}
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
  )
}
