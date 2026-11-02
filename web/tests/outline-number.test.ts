import assert from 'node:assert/strict'
import test from 'node:test'

import { fillParentNumbers, type NumberedNode } from '../src/outline-number.ts'

function node(label: string, number = '', children: NumberedNode[] = []): NumberedNode {
  return { label, heading_numbers: number, children }
}

test('parent inherits the number prefix from its only child', () => {
  const root = node('第二章', '', [node('2.1 环境', '2-1')])

  fillParentNumbers([root])

  assert.equal(root.heading_numbers, '2')
})

test('nested container inherits the correct depth prefix', () => {
  const child = node('2.1.3 部署', '2-1-3')
  const parent = node('2.1 环境', '', [child])
  const root = node('第二章', '', [parent])

  fillParentNumbers([root])

  assert.equal(root.heading_numbers, '2')
  assert.equal(parent.heading_numbers, '2-1')
})

test('an existing number is never overwritten', () => {
  const root = node('第二章', '2', [node('2.1 环境', '2-1')])

  fillParentNumbers([root])

  assert.equal(root.heading_numbers, '2')
})

test('a numeric label can supply the parent number without children', () => {
  const root = node('3.2.1 配置')

  fillParentNumbers([root])

  assert.equal(root.heading_numbers, '3')
})
