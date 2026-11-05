import assert from 'node:assert/strict'
import test from 'node:test'

import { unfiledDocuments } from '../src/document-selection.ts'

test('selection keeps only documents outside folders', () => {
  const rows = [
    { doc_id: 'unfiled-empty', folder: '' },
    { doc_id: 'filed', folder: 'f1' },
    { doc_id: 'unfiled-missing' },
  ]

  assert.deepEqual(
    unfiledDocuments(rows).map((row) => row.doc_id),
    ['unfiled-empty', 'unfiled-missing'],
  )
})

test('selection preserves source order and does not mutate the input', () => {
  const rows = [
    { doc_id: 'b', folder: '' },
    { doc_id: 'a', folder: null },
  ]

  assert.deepEqual(
    unfiledDocuments(rows).map((row) => row.doc_id),
    ['b', 'a'],
  )
  assert.deepEqual(rows, [
    { doc_id: 'b', folder: '' },
    { doc_id: 'a', folder: null },
  ])
})
