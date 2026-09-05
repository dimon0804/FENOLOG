import { useMemo, useSyncExternalStore } from 'react'

/**
 * Тема оформления: светлая, тёмная и «как в системе».
 *
 * Хранилище сделано внешним, а не React-контекстом, ровно по одной причине:
 * тему нужно применить до первой отрисовки, иначе тёмный интерфейс на мгновение
 * мигает светлым. Атрибут на корне документа ставит встроенный скрипт в
 * index.html — он читает тот же ключ, что и этот модуль, и работает до того,
 * как React вообще загрузится. Здесь остаётся только подхватить готовое
 * состояние и уметь его менять.
 *
 * Само оформление живёт в CSS: цвета заданы переменными, а тема выбирает,
 * какой набор переменных подставить. JS цветов не знает — за одним
 * исключением, см. `usePalette` ниже.
 */

// Ключ дублируется во встроенном скрипте index.html. Менять — в обоих местах.
const KEY = 'fenolog-theme'
const MODES = ['light', 'dark', 'system']

const media = window.matchMedia('(prefers-color-scheme: dark)')

let mode = readMode()
let snapshot = `${mode}|${resolveTheme(mode)}`
const listeners = new Set()

function readMode() {
  try {
    const saved = localStorage.getItem(KEY)
    return MODES.includes(saved) ? saved : 'system'
  } catch {
    // В приватном режиме браузера хранилище может быть закрыто целиком.
    // Это не повод падать: сервис просто работает в системной теме.
    return 'system'
  }
}

/** Тема, которая реально нарисована: «как в системе» разворачивается в свою. */
export function resolveTheme(current = mode) {
  if (current !== 'system') return current
  return media.matches ? 'dark' : 'light'
}

/**
 * Тема задаётся атрибутом на корне, а не классом на приложении.
 *
 * Так её видят и элементы вне React — всплывающие подсказки maplibre, которые
 * библиотека вставляет сама, — и встроенный скрипт до загрузки бандла. В режиме
 * «как в системе» атрибут снимается совсем: тогда работает медиавыражение
 * prefers-color-scheme, и тема переключается вместе с системной без участия JS.
 */
function applyMode() {
  const root = document.documentElement
  if (mode === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', mode)
  // Родные элементы браузера — полосы прокрутки, выпадающие списки, поля ввода —
  // красятся не нашими переменными, а этим свойством.
  root.style.colorScheme = mode === 'system' ? 'light dark' : mode
}

function refresh() {
  snapshot = `${mode}|${resolveTheme(mode)}`
  for (const notify of listeners) notify()
}

export function setThemeMode(next) {
  if (!MODES.includes(next) || next === mode) return
  mode = next
  try {
    localStorage.setItem(KEY, next)
  } catch {
    // Не сохранилось — тема продержится до перезагрузки страницы, и только.
  }
  applyMode()
  refresh()
}

// Системная тема меняется на ходу: в Windows это автоматическое переключение по
// времени суток. В режиме «как в системе» интерфейс обязан переключиться следом.
media.addEventListener('change', () => {
  if (mode === 'system') refresh()
})

function subscribe(notify) {
  listeners.add(notify)
  return () => listeners.delete(notify)
}

const getSnapshot = () => snapshot

/** Текущий режим, нарисованная тема и способ её сменить. */
export function useTheme() {
  const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
  const [current, theme] = state.split('|')
  return { mode: current, theme, setMode: setThemeMode }
}

// Цвета, которые нельзя задать стилями.
//
// Recharts рисует оси, сетку и подсказки настройками в JSX, а не классами:
// цвет уходит в атрибут SVG, и переменная CSS туда не подставится. Поэтому
// значения читаются из тех же переменных, что и всё остальное оформление, —
// один источник правды сохраняется, просто путь до него длиннее.
const PALETTE_VARS = {
  ink: '--ink',
  inkSoft: '--ink-soft',
  card: '--card',
  line: '--line',
  grid: '--chart-grid',
  axis: '--chart-axis',
  tick: '--chart-tick',
  series: '--chart-line',
  band: '--chart-band',
  observed: '--chart-dot',
  track: '--donut-track',
  rain: '--chart-rain',
  temp: '--chart-temp',
  green: '--green',
  critical: '--critical',
  suppression: '--suppression',
  compare: '--chart-compare',
}

function readPalette() {
  const style = getComputedStyle(document.documentElement)
  const value = (name) => style.getPropertyValue(name).trim()
  const palette = {}
  for (const [key, name] of Object.entries(PALETTE_VARS)) palette[key] = value(name)
  // Список цветов сравнения приходит одной строкой через запятую: держать пять
  // отдельных переменных ради того же смысла незачем.
  palette.compare = palette.compare.split(',').map((item) => item.trim())
  return palette
}

/** Палитра графиков для текущей темы. Пересчитывается только при её смене. */
export function usePalette() {
  const { theme } = useTheme()
  return useMemo(() => readPalette(), [theme])
}

// Тему мог применить встроенный скрипт, но если его почему-то нет (открыт
// собранный index.html без него), применим сами — до первой отрисовки React.
applyMode()
