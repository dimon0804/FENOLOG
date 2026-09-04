import { useState } from 'react'

import { formatDate } from '../dict.js'

/**
 * Левая колонка: поиск региона и список сохранённых участков.
 *
 * Поиск стоит первым не случайно — сценарий постановки начинается с того, что
 * пользователь указывает интересующий регион.
 */
export default function Sidebar({
  polygons,
  selectedId,
  onSelect,
  onRename,
  onDelete,
  onSearch,
  places,
  searching,
  onPickPlace,
  searchNote,
}) {
  const [query, setQuery] = useState('')

  return (
    <aside className="sidebar">
      <div className="block">
        <h2>Регион</h2>
        <form
          onSubmit={(event) => {
            event.preventDefault()
            onSearch(query)
          }}
          className="stack"
        >
          <input
            type="search"
            placeholder="Сальский район, Кубань, Аксай…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button className="primary" type="submit" disabled={searching || query.trim().length < 2}>
            {searching ? 'Ищу…' : 'Найти на карте'}
          </button>
        </form>

        {searchNote && <p className="small muted" style={{ marginBottom: 0 }}>{searchNote}</p>}

        {places?.length > 0 && (
          <div className="stack" style={{ marginTop: 10 }}>
            {places.map((place) => (
              <div
                key={`${place.name}-${place.center.join()}`}
                className="polygon-item"
                onClick={() => onPickPlace(place)}
              >
                <div className="small">{place.name}</div>
                {place.type && <div className="meta">{place.type}</div>}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="block">
        <h2>Сохранённые участки {polygons.length > 0 && `(${polygons.length})`}</h2>
        {polygons.length === 0 && (
          <p className="small muted">
            Пока пусто. Выберите контур на карте или нарисуйте свой — и сохраните его,
            чтобы вернуться к полю позже.
          </p>
        )}
        {polygons.map((polygon) => (
          <PolygonItem
            key={polygon.id}
            polygon={polygon}
            active={polygon.id === selectedId}
            onSelect={onSelect}
            onRename={onRename}
            onDelete={onDelete}
          />
        ))}
      </div>
    </aside>
  )
}

function PolygonItem({ polygon, active, onSelect, onRename, onDelete }) {
  const [editing, setEditing] = useState(false)
  const [name, setName] = useState(polygon.name)

  if (editing) {
    return (
      <form
        className="polygon-item"
        onSubmit={(event) => {
          event.preventDefault()
          onRename(polygon.id, name)
          setEditing(false)
        }}
      >
        <input type="text" value={name} autoFocus onChange={(e) => setName(e.target.value)} />
        <div className="actions">
          <button className="ghost" type="submit">Сохранить</button>
          <button className="ghost" type="button" onClick={() => { setName(polygon.name); setEditing(false) }}>
            Отмена
          </button>
        </div>
      </form>
    )
  }

  return (
    <div className={`polygon-item${active ? ' active' : ''}`} onClick={() => onSelect(polygon)}>
      <div className="name">{polygon.name}</div>
      <div className="meta">
        {polygon.area_ha} га
        {polygon.crop_type ? ` · ${polygon.crop_type}` : ''}
        {polygon.source === 'osm' ? ' · из OSM' : ' · нарисован'}
      </div>
      <div className="meta">
        {polygon.last_analyzed_at
          ? `анализ от ${formatDate(polygon.last_analyzed_at)}`
          : 'ещё не анализировался'}
      </div>
      <div className="actions" onClick={(event) => event.stopPropagation()}>
        <button className="ghost small" onClick={() => setEditing(true)}>Переименовать</button>
        <button className="ghost small danger" onClick={() => onDelete(polygon)}>Удалить</button>
      </div>
    </div>
  )
}
