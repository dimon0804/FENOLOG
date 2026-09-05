import { useMemo } from 'react'

// Тот же слой снимков, что и подложка «Снимок» на карте. Один источник на две
// задачи: если он отвалится, это будет видно и там, и там, а не в одном месте.
const TILES = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile'
const TILE = 256

/**
 * Спутниковая миниатюра участка.
 *
 * В макете у каждой строки таблицы стоит картинка поля. Брать её неоткуда —
 * скриншотов полей сервис не хранит, — поэтому миниатюра собирается из тех же
 * тайлов, которыми рисуется карта: это настоящий снимок именно этого места, а
 * не иллюстрация из набора.
 *
 * Почему четыре тайла, а не один. Центр поля попадает в произвольную точку
 * тайла, и если взять только его, у половины участков поле окажется у самого
 * края миниатюры, а вторую половину кадра займёт пустота. Берём квадрат 2×2,
 * внутри которого центр гарантированно не ближе половины тайла к краю, и
 * сдвигаем его так, чтобы центр поля пришёлся на середину миниатюры.
 *
 * Масштаб считается от площади: у поля в 60 гектаров и у поля в 6000 при одном
 * зуме на снимке видно совершенно разное, и мелкое превращается в точку.
 */
export default function Thumb({ center, areaHa, size = 54 }) {
  const view = useMemo(() => plan(center, areaHa, size), [center, areaHa, size])

  if (!view) {
    return <span className="thumb empty" style={{ width: size, height: size }} />
  }

  return (
    <span className="thumb" style={{ width: size, height: size }}>
      <span
        className="thumb-grid"
        style={{ transform: `translate(${view.dx}px, ${view.dy}px)` }}
      >
        {view.tiles.map((tile) => (
          <img
            key={`${tile.x}-${tile.y}`}
            src={`${TILES}/${view.z}/${tile.y}/${tile.x}`}
            alt=""
            loading="lazy"
            width={TILE}
            height={TILE}
            style={{ left: tile.left, top: tile.top }}
          />
        ))}
      </span>
    </span>
  )
}

function plan(center, areaHa, size) {
  if (!Array.isArray(center) || center.length !== 2) return null
  const [lon, lat] = center
  if (!Number.isFinite(lon) || !Number.isFinite(lat) || Math.abs(lat) > 85) return null

  // Сторона поля, если считать его квадратным. Точность здесь не нужна: от
  // числа зависит только выбор зума, а зум всё равно целый.
  const side = Math.sqrt(Math.max(areaHa || 1, 0.01) * 10000)
  // Хотим, чтобы поле занимало примерно половину миниатюры.
  const metersPerPixel = side / (size * 0.55)
  const scale = (156543.03392 * Math.cos((lat * Math.PI) / 180)) / metersPerPixel
  const z = Math.max(9, Math.min(17, Math.round(Math.log2(scale))))

  const n = 2 ** z
  const x = ((lon + 180) / 360) * n
  const latRad = (lat * Math.PI) / 180
  const y = ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n

  // Левый верхний тайл квадрата 2×2 выбираем так, чтобы центр оказался внутри
  // средней половины: тогда до любого края не меньше половины тайла.
  const x0 = Math.floor(x - 0.5)
  const y0 = Math.floor(y - 0.5)

  const tiles = []
  for (let dy = 0; dy < 2; dy += 1) {
    for (let dx = 0; dx < 2; dx += 1) {
      tiles.push({ x: x0 + dx, y: y0 + dy, left: dx * TILE, top: dy * TILE })
    }
  }

  return {
    z,
    tiles,
    dx: Math.round(size / 2 - (x - x0) * TILE),
    dy: Math.round(size / 2 - (y - y0) * TILE),
  }
}
