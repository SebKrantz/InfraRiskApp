// A small renderer for the assistant's constrained markdown subset: headings,
// paragraphs, bullet/numbered lists, fenced + inline code, bold, italic, links
// and pipe tables. No dependency, no HTML passthrough.

import { Fragment, type ReactNode } from 'react'

function inline(text: string, keyBase: string): ReactNode[] {
  // tokenize `code`, **bold**, *italic*, [label](url)
  const out: ReactNode[] = []
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)|(\[[^\]]+\]\((?:https?:\/\/|\/)[^\s)]+\))/g
  let pos = 0
  let m: RegExpExecArray | null
  let i = 0
  while ((m = re.exec(text)) !== null) {
    if (m.index > pos) out.push(text.slice(pos, m.index))
    const tok = m[0]
    const key = `${keyBase}-${i++}`
    if (tok.startsWith('`'))
      out.push(
        <code key={key} className="rounded bg-gray-700/70 px-1 py-0.5 font-mono text-[11px]">
          {tok.slice(1, -1)}
        </code>,
      )
    else if (tok.startsWith('**')) out.push(<strong key={key}>{tok.slice(2, -2)}</strong>)
    else if (tok.startsWith('*')) out.push(<em key={key}>{tok.slice(1, -1)}</em>)
    else {
      const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(tok)
      if (link)
        out.push(
          <a
            key={key}
            href={link[2]}
            target="_blank"
            rel="noreferrer"
            className="text-blue-400 underline decoration-blue-400/40 hover:text-blue-300"
          >
            {link[1]}
          </a>,
        )
      else out.push(tok)
    }
    pos = m.index + tok.length
  }
  if (pos < text.length) out.push(text.slice(pos))
  return out
}

function table(lines: string[], key: string): ReactNode {
  const rows = lines.map((l) =>
    l
      .replace(/^\s*\|/, '')
      .replace(/\|\s*$/, '')
      .split('|')
      .map((c) => c.trim()),
  )
  const [head, ...body] = rows
  const cells = body.filter((r) => !r.every((c) => /^:?-{2,}:?$/.test(c)))
  return (
    <div key={key} className="my-1.5 overflow-x-auto">
      <table className="min-w-full border-collapse text-[11px]">
        <thead>
          <tr>
            {head.map((c, i) => (
              <th
                key={i}
                className="border border-gray-700 bg-gray-800 px-2 py-1 text-left font-semibold"
              >
                {inline(c, `${key}-h${i}`)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cells.map((r, ri) => (
            <tr key={ri}>
              {r.map((c, ci) => (
                <td key={ci} className="border border-gray-700/60 px-2 py-1 align-top">
                  {inline(c, `${key}-${ri}-${ci}`)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Markdown({ text }: { text: string }) {
  const lines = text.split('\n')
  const blocks: ReactNode[] = []
  let i = 0
  let k = 0
  while (i < lines.length) {
    const line = lines[i]
    const key = `b${k++}`
    if (line.startsWith('```')) {
      const code: string[] = []
      i++
      while (i < lines.length && !lines[i].startsWith('```')) code.push(lines[i++])
      i++ // closing fence
      blocks.push(
        <pre
          key={key}
          className="my-1.5 overflow-x-auto rounded-md bg-gray-800 p-2 font-mono text-[11px] leading-snug"
        >
          {code.join('\n')}
        </pre>,
      )
      continue
    }
    const heading = /^(#{1,3})\s+(.*)$/.exec(line)
    if (heading) {
      const sizes = ['text-sm font-bold', 'text-[13px] font-bold', 'text-xs font-bold']
      blocks.push(
        <div key={key} className={`mt-2 ${sizes[heading[1].length - 1]}`}>
          {inline(heading[2], key)}
        </div>,
      )
      i++
      continue
    }
    if (/^\s*\|.*\|\s*$/.test(line)) {
      const tbl: string[] = []
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) tbl.push(lines[i++])
      blocks.push(table(tbl, key))
      continue
    }
    const bullet = /^\s*[-*]\s+(.*)$/.exec(line)
    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line)
    if (bullet || numbered) {
      const ordered = !!numbered
      const listItems: string[] = []
      while (i < lines.length) {
        const bm = /^\s*[-*]\s+(.*)$/.exec(lines[i])
        const nm = /^\s*\d+[.)]\s+(.*)$/.exec(lines[i])
        const match = ordered ? nm : bm
        if (!match) break
        listItems.push(match[1])
        i++
      }
      const cls = 'my-1 space-y-0.5 pl-4'
      blocks.push(
        ordered ? (
          <ol key={key} className={`${cls} list-decimal`}>
            {listItems.map((it, j) => (
              <li key={j}>{inline(it, `${key}-${j}`)}</li>
            ))}
          </ol>
        ) : (
          <ul key={key} className={`${cls} list-disc`}>
            {listItems.map((it, j) => (
              <li key={j}>{inline(it, `${key}-${j}`)}</li>
            ))}
          </ul>
        ),
      )
      continue
    }
    if (line.trim() === '') {
      i++
      continue
    }
    // paragraph: consume consecutive plain lines
    const para: string[] = [line]
    i++
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !/^(#{1,3})\s|^```|^\s*[-*]\s|^\s*\d+[.)]\s|^\s*\|.*\|\s*$/.test(lines[i])
    )
      para.push(lines[i++])
    blocks.push(
      <p key={key} className="my-1 leading-relaxed">
        {para.map((l, j) => (
          <Fragment key={j}>
            {j > 0 && ' '}
            {inline(l, `${key}-${j}`)}
          </Fragment>
        ))}
      </p>,
    )
  }
  return <div className="text-xs text-gray-200">{blocks}</div>
}
