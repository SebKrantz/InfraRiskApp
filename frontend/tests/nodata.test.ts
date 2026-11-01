// The sidebar's no-data line (src/lib/nodata.ts).
// No test runner in this repo: Node runs it directly (type stripping), from frontend/:
//   node --test tests/*.test.ts
import assert from 'node:assert/strict'
import { test } from 'node:test'

import { nodataLine } from '../src/lib/nodata.ts'

test('a result without no-data says nothing', () => {
  assert.equal(nodataLine({ nodata_samples: 0, nodata_share_pct: 0 }, 'LineString'), null)
  assert.equal(nodataLine({}, 'Point'), null)
})

test('a line gives the share of samples, the length and what no data means', () => {
  const line = nodataLine(
    {
      nodata_samples: 9186,
      nodata_share_pct: 100,
      nodata_km: 918.6,
      nodata_note: 'no data = below 49 cm/s² (the layer\'s minimum), under the threshold of 98; treated as not exposed',
    },
    'LineString'
  )
  assert.equal(
    line,
    'No data at 100% of sample points (918.6 km). No data = below 49 cm/s² (the layer\'s minimum), under the threshold of 98; treated as not exposed.'
  )
})

test('points count assets and carry no length', () => {
  const line = nodataLine({ nodata_samples: 3, nodata_share_pct: 2.5, nodata_note: 'no data — hazard unknown here; counted as not exposed' }, 'Point')
  assert.equal(line, 'No data at 2.5% of assets. No data — hazard unknown here; counted as not exposed.')
})
