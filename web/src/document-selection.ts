/** 全选只返回尚未归入目录的文档；空串和缺失都表示“未分类”。 */
export function unfiledDocuments<T extends { folder?: string | null }>(
  rows: readonly T[],
): T[] {
  return rows.filter((row) => !row.folder)
}
