/**
 * 章节树编号补全。
 *
 * 后端按「实际含块的叶子路径」汇总；由更深路径带出来的父节点没有自己的
 * heading_numbers。这里按“自身 → 标题前缀 → 子节点前缀”的顺序补全。
 */
export interface NumberedNode {
  label: string
  heading_numbers: string
  children: NumberedNode[]
}

const LEADING_NUMBER = /^[\s·•]*(\d+(?:[.．]\d+)*)/

function prefixAtDepth(value: string, depth: number): string {
  return value.split('-').slice(0, depth + 1).join('-')
}

function numberFromLabel(label: string, depth: number): string {
  const match = label.match(LEADING_NUMBER)
  if (!match) return ''
  return match[1].replaceAll('．', '.').split('.').slice(0, depth + 1).join('-')
}

/** 原地补全树中的空编号；已有编号不改。 */
export function fillParentNumbers(nodes: NumberedNode[], depth = 0): void {
  for (const node of nodes) {
    fillParentNumbers(node.children as NumberedNode[], depth + 1)
    if (node.heading_numbers) continue

    const fromLabel = numberFromLabel(node.label, depth)
    if (fromLabel && fromLabel.split('-').length === depth + 1) {
      node.heading_numbers = fromLabel
      continue
    }

    const childNumber = node.children
      .map((child) => child.heading_numbers)
      .find((value) => value.split('-').length === depth + 2)
    if (childNumber) node.heading_numbers = prefixAtDepth(childNumber, depth)
  }
}
