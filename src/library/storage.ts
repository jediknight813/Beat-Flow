export type SongDetails = {
  id: string
  title: string
  artist: string
  createdAt: number
  updatedAt: number
  duration: number
  bpm: number
  charts: { difficulty: string; notes: number }[]
  cover: Blob | null
  coverSource: 'detected' | 'generated' | 'custom'
  fileName: string
}

export type SavedSong = { details: SongDetails; zip: Blob }

let database: Promise<IDBDatabase> | null = null
function openDatabase(): Promise<IDBDatabase> {
  if (database) return database
  database = new Promise((resolve, reject) => {
    const request = indexedDB.open('beatflow-song-history', 1)
    request.onupgradeneeded = () => {
      request.result.createObjectStore('songs', { keyPath: 'id' })
      request.result.createObjectStore('archives')
    }
    request.onerror = () => { database = null; reject(request.error) }
    request.onsuccess = () => {
      const db = request.result
      db.onversionchange = () => { db.close(); database = null }
      resolve(db)
    }
  })
  return database
}

function complete(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve()
    transaction.onabort = () => reject(transaction.error ?? new Error('History storage was interrupted.'))
    transaction.onerror = () => reject(transaction.error ?? new Error('Unable to save song history.'))
  })
}

export async function listSongs(): Promise<SongDetails[]> {
  const db = await openDatabase()
  return new Promise((resolve, reject) => {
    // Archives live in a separate store so the list never loads every ZIP.
    const request = db.transaction('songs').objectStore('songs').getAll()
    request.onsuccess = () => resolve((request.result as SongDetails[]).sort((a, b) => b.createdAt - a.createdAt))
    request.onerror = () => reject(request.error)
  })
}

export async function getSong(id: string): Promise<SavedSong> {
  const db = await openDatabase()
  const transaction = db.transaction(['songs', 'archives'])
  const done = complete(transaction)
  const details = transaction.objectStore('songs').get(id)
  const zip = transaction.objectStore('archives').get(id)
  await done
  if (!details.result || !zip.result) throw new Error('This saved map is no longer available.')
  return { details: details.result as SongDetails, zip: zip.result as Blob }
}

export async function saveSong(song: SavedSong): Promise<void> {
  const db = await openDatabase()
  const transaction = db.transaction(['songs', 'archives'], 'readwrite')
  const done = complete(transaction)
  transaction.objectStore('songs').put(song.details)
  transaction.objectStore('archives').put(song.zip, song.details.id)
  await done
}

export async function deleteSong(id: string): Promise<void> {
  const db = await openDatabase()
  const transaction = db.transaction(['songs', 'archives'], 'readwrite')
  const done = complete(transaction)
  transaction.objectStore('songs').delete(id)
  transaction.objectStore('archives').delete(id)
  await done
}

export async function clearSongHistory(): Promise<void> {
  const db = await openDatabase()
  const transaction = db.transaction(['songs', 'archives'], 'readwrite')
  const done = complete(transaction)
  transaction.objectStore('songs').clear()
  transaction.objectStore('archives').clear()
  await done
}
