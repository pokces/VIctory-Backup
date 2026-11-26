//! Compare the existing compaction kernel and a full readback on one DIRECT queue.
use super::*;
use std::time::Instant;

const PAGES: usize = 512;
const SIZE: usize = PAGES * PAGE;
const RESULT_SIZE: usize = HEADER_BYTES + PAGES * RECORD_BYTES;
const SAMPLES: usize = 8;

fn local(f: &Fixture, size: usize, flags: D3D12_RESOURCE_FLAGS) -> Result<ID3D12Resource> {
    buffer(
        &f.device,
        size,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        flags,
    )
}

fn readback(f: &Fixture, size: usize) -> Result<ID3D12Resource> {
    buffer(
        &f.device,
        size,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE,
    )
}

fn submit(f: &Fixture, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> Result<()> {
    // SAFETY: each call owns its list and allocator until its queue fence completes.
    unsafe {
        let alloc: ID3D12CommandAllocator = f
            .device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList =
            f.device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
        record(&list);
        list.Close()?;
        let fence: ID3D12Fence = f.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        f.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
        f.queue.Signal(&fence, 1)?;
        fence.SetEventOnCompletion(1, HANDLE::default())?;
        f.device.GetDeviceRemovedReason()
    }
}

fn bytes(resource: &ID3D12Resource, size: usize) -> Result<Vec<u8>> {
    // SAFETY: the requested prefix is within the readback resource and submission has finished.
    unsafe {
        let mut mapped = ptr::null_mut();
        resource.Map(
            0,
            Some(&D3D12_RANGE {
                Begin: 0,
                End: size,
            }),
            Some(&mut mapped),
        )?;
        let data = std::slice::from_raw_parts(mapped.cast::<u8>(), size).to_vec();
        resource.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        Ok(data)
    }
}

struct Timer {
    heap: ID3D12QueryHeap,
    ticks: ID3D12Resource,
    hz: u64,
}

impl Timer {
    fn new(f: &Fixture) -> Result<Self> {
        let mut heap = None;
        // SAFETY: the heap has two slots and the resolve buffer has two u64s.
        unsafe {
            f.device.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2,
                    NodeMask: 0,
                },
                &mut heap,
            )?;
            Ok(Self {
                heap: heap.unwrap(),
                ticks: readback(f, 16)?,
                hz: f.queue.GetTimestampFrequency()?,
            })
        }
    }

    fn time(&self, f: &Fixture, emit: impl FnOnce(&ID3D12GraphicsCommandList)) -> Result<f64> {
        // SAFETY: both timestamp queries are issued and resolved before the fence is observed.
        submit(f, |list| unsafe {
            list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            emit(list);
            list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.ticks, 0);
        })?;
        let ticks = bytes(&self.ticks, 16)?;
        let begin = u64::from_le_bytes(ticks[..8].try_into().unwrap());
        let end = u64::from_le_bytes(ticks[8..].try_into().unwrap());
        assert!(end >= begin && self.hz > 0);
        Ok((end - begin) as f64 * 1000.0 / self.hz as f64)
    }
}

struct Buffers {
    before: ID3D12Resource,
    after: ID3D12Resource,
    result: ID3D12Resource,
    poison: ID3D12Resource,
    download: ID3D12Resource,
}

impl Buffers {
    fn new(f: &Fixture) -> Result<Self> {
        let mut initial = vec![GUARD; RESULT_SIZE];
        initial[..8].fill(0);
        let seed = upload(&f.device, &initial)?;
        let poison = local(f, RESULT_SIZE, D3D12_RESOURCE_FLAG_NONE)?;
        // SAFETY: source and destination both span RESULT_SIZE bytes.
        submit(f, |list| unsafe {
            list.CopyBufferRegion(&poison, 0, &seed, 0, RESULT_SIZE as u64)
        })?;
        Ok(Self {
            before: local(f, SIZE, D3D12_RESOURCE_FLAG_NONE)?,
            after: local(f, SIZE, D3D12_RESOURCE_FLAG_NONE)?,
            result: local(f, RESULT_SIZE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)?,
            poison,
            download: readback(f, RESULT_SIZE)?,
        })
    }

    fn compare(&self, f: &Fixture, list: &ID3D12GraphicsCommandList, download_bytes: usize) {
        let constants = [0u32, SIZE as u32, 0, PAGES as u32];
        // SAFETY: buffers decay to COMMON between queue submissions. The SRVs
        // promote on read; the explicit UAV-to-copy transition orders output writes.
        unsafe {
            list.ResourceBarrier(&[transition(
                &self.result,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            list.SetComputeRootSignature(&f.root);
            list.SetPipelineState(&f.pipeline);
            list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, self.before.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(2, self.after.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, self.result.GetGPUVirtualAddress());
            list.Dispatch(PAGES as u32, 1, 1);
            list.ResourceBarrier(&[transition(
                &self.result,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyBufferRegion(&self.download, 0, &self.result, 0, download_bytes as u64);
        }
    }
}

fn verify_output(data: &[u8], expected: &BTreeMap<u32, Vec<u8>>) {
    assert_eq!(read_word(data, 0) as usize, expected.len());
    assert_eq!(read_word(data, 4), 0, "unexpected capacity overflow");
    let mut unseen = expected.clone();
    let (records, tail) = data[HEADER_BYTES..].as_chunks::<RECORD_BYTES>();
    assert!(tail.is_empty());
    for record in records {
        let id = read_word(record, 0);
        let oracle = unseen.remove(&id).expect("unique expected page");
        assert_bytes("readback record", record, &oracle, 0);
    }
    assert!(unseen.is_empty());
}

pub(super) fn measure(f: &Fixture) -> Result<()> {
    let b = Buffers::new(f)?;
    let timer = Timer::new(f)?;
    // Deterministic input; changing one byte at a seeded offset in each selected
    // page varies the number of dirty pages without changing the mutation density per page.
    let mut seed = 0x73109a4du32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let before: Vec<u8> = (0..SIZE / 4).flat_map(|_| next().to_le_bytes()).collect();
    let mut pages: Vec<usize> = (0..PAGES).collect();
    for i in (1..PAGES).rev() {
        pages.swap(i, next() as usize % (i + 1));
    }
    let positions: Vec<_> = pages
        .iter()
        .map(|p| p * PAGE + next() as usize % PAGE)
        .collect();
    let a = upload(&f.device, &before)?;
    // SAFETY: source and destination have SIZE bytes.
    submit(f, |list| unsafe {
        list.CopyBufferRegion(&b.before, 0, &a, 0, SIZE as u64)
    })?;
    eprintln!(
        "timestamp frequency {} Hz; before/after {} bytes; record {} bytes",
        timer.hz, SIZE, RECORD_BYTES
    );
    for count in [5, 256, 384, 448, 512] {
        let mut after = before.clone();
        for &p in &positions[..count] {
            after[p] ^= 0x6d;
        }
        let case = Case {
            name: "timestamp fixture",
            range: 0..SIZE,
            bias: 0,
            capacity: PAGES,
            changes: Vec::new(),
        };
        let expected = expect_records(&case, &before, &after);
        assert_eq!(expected.len(), count);
        let compact = HEADER_BYTES + count * RECORD_BYTES;
        let src = upload(&f.device, &after)?;
        // SAFETY: source and destination have SIZE bytes.
        submit(f, |list| unsafe {
            list.CopyBufferRegion(&b.after, 0, &src, 0, SIZE as u64)
        })?;
        for iteration in 0..SAMPLES {
            // Rotate order within each density. Each method gets one warm-up and seven samples.
            let modes = ["full", "presized", "discover"];
            for i in 0..modes.len() {
                let mode = modes[(i + iteration) % modes.len()];
                // SAFETY: both local buffers hold RESULT_SIZE bytes. Every variant,
                // including the control, starts after the same untimed reset.
                submit(f, |list| unsafe {
                    list.CopyBufferRegion(&b.result, 0, &b.poison, 0, RESULT_SIZE as u64)
                })?;
                let wall = Instant::now();
                let (gpu_ms, wire_bytes, full) = match mode {
                    "full" => {
                        // SAFETY: both buffers have room for the complete input.
                        let ms = timer.time(f, |list| unsafe {
                            list.CopyBufferRegion(&b.download, 0, &b.after, 0, SIZE as u64)
                        })?;
                        (ms, SIZE, true)
                    }
                    "presized" => (
                        timer.time(f, |list| b.compare(f, list, compact))?,
                        compact,
                        false,
                    ),
                    _ => {
                        let head_ms = timer.time(f, |list| b.compare(f, list, HEADER_BYTES))?;
                        let header = bytes(&b.download, HEADER_BYTES)?;
                        let found = read_word(&header, 0) as usize;
                        let size = HEADER_BYTES + found * RECORD_BYTES;
                        assert_eq!(found, count);
                        assert_eq!(read_word(&header, 4), 0);
                        // Size now comes from the GPU. For a dense batch, read the full after-buffer.
                        let full = size >= SIZE;
                        let payload = if full { SIZE } else { size - HEADER_BYTES };
                        let (src, offset) = if full {
                            (&b.after, 0)
                        } else {
                            (&b.result, HEADER_BYTES as u64)
                        };
                        // SAFETY: the count matched the bounded input. The compact
                        // copy follows the header; the full copy starts at zero.
                        let copy_ms = timer.time(f, |list| unsafe {
                            list.CopyBufferRegion(&b.download, offset, src, offset, payload as u64)
                        })?;
                        (head_ms + copy_ms, HEADER_BYTES + payload, full)
                    }
                };
                let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
                // Validate once before timing samples and again at the end, outside the timer.
                if iteration == 0 || iteration + 1 == SAMPLES {
                    if full {
                        assert_bytes("full readback", &bytes(&b.download, SIZE)?, &after, 0);
                    } else {
                        verify_output(&bytes(&b.download, compact)?, &expected);
                    }
                }
                eprintln!(
                    "TIMING,{count},{iteration},{mode},{wire_bytes},{gpu_ms:.6},{wall_ms:.6}"
                );
            }
        }
    }
    Ok(())
}
