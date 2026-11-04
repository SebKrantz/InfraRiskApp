// The Export menu's client calls (src/lib/assistantApi.ts). No test runner in this repo:
// Node runs it directly (type stripping), from frontend/:  node --test tests/
import assert from 'node:assert/strict'
import { test } from 'node:test'

const requested: string[] = []
let reply: { status: number; body: string } = { status: 200, body: '' }
Object.assign(globalThis, {
  fetch: async (url: string) => {
    requested.push(url)
    return new Response(reply.body, { status: reply.status })
  },
})

const { exportMarkdown, exportUrl } = await import('../src/lib/assistantApi.ts')

test('the download links name the conversation and the format', () => {
  assert.equal(
    exportUrl('abc123', 'md'),
    '/api/assistant/conversations/abc123/export?format=md',
  )
  assert.equal(
    exportUrl('abc123', 'json'),
    '/api/assistant/conversations/abc123/export?format=json',
  )
  // an id the browser or an API client made up cannot escape the path
  assert.equal(
    exportUrl('a/b?c', 'md'),
    '/api/assistant/conversations/a%2Fb%3Fc/export?format=md',
  )
})

test('the Markdown comes back as text, fetched from the md link', async () => {
  reply = { status: 200, body: '# Assistant conversation\n' }
  assert.equal(await exportMarkdown('abc123'), '# Assistant conversation\n')
  assert.equal(requested.at(-1), exportUrl('abc123', 'md'))
})

test("a refusal throws the server's reason, or the status when there is none", async () => {
  reply = { status: 404, body: JSON.stringify({ detail: 'conversation not found (it was cleared)' }) }
  await assert.rejects(exportMarkdown('gone'), /conversation not found \(it was cleared\)/)
  reply = { status: 502, body: '<html>bad gateway</html>' }
  await assert.rejects(exportMarkdown('gone'), /export failed \(502\)/)
})
