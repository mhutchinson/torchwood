// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mpt

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"slices"
	"testing"
	"time"

	"filippo.io/torchwood/mpt/internal/pmem"
)

type benchSyncCtl struct {
	deferSync bool
	files     []*benchCountingFile
}

func (c *benchSyncCtl) syncPending() error {
	for _, df := range c.files {
		if df.pendingSync || df.dirty {
			df.fsyncCalls++
			if err := df.f.Sync(); err != nil {
				return err
			}
			df.pendingSync = false
			df.dirty = false
		}
	}
	return nil
}

func (c *benchSyncCtl) resetCounters() {
	for _, df := range c.files {
		df.readCalls = 0
		df.readBytes = 0
		df.writeCalls = 0
		df.writeBytes = 0
		df.fsyncCalls = 0
	}
}

type ioCounters struct {
	treeReadCalls  int64
	treeReadBytes  int64
	treeWriteCalls int64
	treeWriteBytes int64
	leafReadCalls  int64
	leafReadBytes  int64
	leafWriteCalls int64
	leafWriteBytes int64
	fsyncCalls     int64
}

func (c *benchSyncCtl) snapshotCounters() ioCounters {
	// files order: [dfLeaf, dfTree1, dfTree2]
	leaf, t1, t2 := c.files[0], c.files[1], c.files[2]
	return ioCounters{
		treeReadCalls:  t1.readCalls + t2.readCalls,
		treeReadBytes:  t1.readBytes + t2.readBytes,
		treeWriteCalls: t1.writeCalls + t2.writeCalls,
		treeWriteBytes: t1.writeBytes + t2.writeBytes,
		leafReadCalls:  leaf.readCalls,
		leafReadBytes:  leaf.readBytes,
		leafWriteCalls: leaf.writeCalls,
		leafWriteBytes: leaf.writeBytes,
		fsyncCalls:     leaf.fsyncCalls + t1.fsyncCalls + t2.fsyncCalls,
	}
}

type benchCountingFile struct {
	f           *os.File
	ctl         *benchSyncCtl
	size        int64
	pendingSync bool
	dirty       bool

	readCalls  int64
	readBytes  int64
	writeCalls int64
	writeBytes int64
	fsyncCalls int64
}

func (d *benchCountingFile) Fd() uintptr {
	return d.f.Fd()
}

func (d *benchCountingFile) ReadAt(b []byte, off int64) (int, error) {
	d.readCalls++
	n, err := d.f.ReadAt(b, off)
	d.readBytes += int64(n)
	return n, err
}

func (d *benchCountingFile) WriteAt(b []byte, off int64) (int, error) {
	if d.ctl != nil && (d.pendingSync || off < d.size) {
		for _, peer := range d.ctl.files {
			if peer.pendingSync {
				peer.fsyncCalls++
				if err := peer.f.Sync(); err != nil {
					return 0, err
				}
				peer.pendingSync = false
				peer.dirty = false
			}
		}
	}
	d.dirty = true
	d.writeCalls++
	n, err := d.f.WriteAt(b, off)
	d.writeBytes += int64(n)
	if end := off + int64(n); end > d.size {
		d.size = end
	}
	return n, err
}

func (d *benchCountingFile) Sync() error {
	if d.ctl != nil && d.ctl.deferSync {
		d.pendingSync = true
		return nil
	}
	d.fsyncCalls++
	if err := d.f.Sync(); err != nil {
		return err
	}
	d.pendingSync = false
	d.dirty = false
	return nil
}

func (d *benchCountingFile) Close() error {
	return d.f.Close()
}

func openBenchDiskTree(dir string, create bool, opts ...Option) (Tree, *benchSyncCtl, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, nil, err
	}
	paths := [3]string{
		filepath.Join(dir, "mpt.tree1"),
		filepath.Join(dir, "mpt.tree2"),
		filepath.Join(dir, "mpt.disk"),
	}
	flag := os.O_RDWR
	if create {
		flag = os.O_RDWR | os.O_CREATE | os.O_EXCL
	}
	var files [3]*os.File
	var sizes [3]int64
	for i, p := range paths {
		if !create {
			st, err := os.Stat(p)
			if err != nil {
				return nil, nil, err
			}
			sizes[i] = st.Size()
		}
		f, err := os.OpenFile(p, flag, 0o666)
		if err != nil {
			for j := range i {
				files[j].Close()
			}
			return nil, nil, err
		}
		files[i] = f
	}
	ctl := &benchSyncCtl{}
	df1 := &benchCountingFile{f: files[0], ctl: ctl, size: sizes[0]}
	df2 := &benchCountingFile{f: files[1], ctl: ctl, size: sizes[1]}
	df3 := &benchCountingFile{f: files[2], ctl: ctl, size: sizes[2]}
	ctl.files = []*benchCountingFile{df3, df1, df2}

	tr, err := New(df1, df2, df3, opts...)
	if err != nil {
		for _, f := range files {
			f.Close()
		}
		return nil, nil, err
	}
	if create {
		if _, err := tr.Snap(0); err != nil {
			tr.Close()
			return nil, nil, err
		}
		if err := tr.Sync(); err != nil {
			tr.Close()
			return nil, nil, err
		}
		if err := ctl.syncPending(); err != nil {
			tr.Close()
			return nil, nil, err
		}
	}
	return tr, ctl, nil
}

var benchTLDs = []string{
	"com", "com", "com", "com", "net", "org", "io", "dev", "de", "uk.co", "app", "cloud",
}

var benchSubPrefixes = []string{
	"www", "api", "cdn", "mail", "auth", "app", "edge", "portal",
}

// generateReversedDNSBatch generates n unique realistic reversed-DNS keys (~28B)
// with 32-byte values, sorted strictly by Key.Compare.
func generateReversedDNSBatch(seed uint64, startID, count int, valEpoch uint64) []KeyVal {
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	kvs := make([]KeyVal, count)
	// Pack keys and values contiguously to minimize fixture GC pressure.
	valArena := make([]byte, count*32)
	keyArena := make([]byte, 0, count*32)

	// ~8 subdomains per registrable 2LD on average, matching multi-SAN/subdomain CT distributions.
	numDomains := max(256, (startID+count)/8)

	for i := range count {
		id := startID + i
		tld := benchTLDs[id%len(benchTLDs)]
		domID := r.IntN(numDomains)
		sub := benchSubPrefixes[(id/3)%len(benchSubPrefixes)]

		startOff := len(keyArena)
		// Format: <tld>.d<6-hex-dom>.<sub>-<6-hex-id> -> ~26-30 bytes
		keyArena = fmt.Appendf(keyArena, "%s.dom%05x.%s-%06x", tld, domID, sub, id)
		k := Key(keyArena[startOff:len(keyArena):len(keyArena)])

		var vb [16]byte
		binary.LittleEndian.PutUint64(vb[:8], uint64(id))
		binary.LittleEndian.PutUint64(vb[8:], valEpoch)
		sum := sha256.Sum256(vb[:])
		v := Val(valArena[i*32 : (i+1)*32 : (i+1)*32])
		copy(v, sum[:])

		kvs[i] = KeyVal{Key: k, Val: v}
	}
	slices.SortFunc(kvs, KeyVal.CompareKey)
	return kvs
}

func BenchmarkSetVsSetBatch10K(b *testing.B) {
	batch := generateReversedDNSBatch(42, 0, 10_000, 1)

	for _, tc := range []struct {
		name     string
		useBatch bool
		opts     []Option
	}{
		{"A_Set_StdIO", false, nil},
		{"B_SetBatch_StdIO", true, nil},
		{"C_Set_BufMmapIO", false, []Option{BufferLeafIO(true)}},
		{"D_SetBatch_BufMmapIO", true, []Option{BufferLeafIO(true)}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				b.StopTimer()
				dir := b.TempDir()
				tr, ctl, err := openBenchDiskTree(dir, true, tc.opts...)
				if err != nil {
					b.Fatal(err)
				}
				ctl.resetCounters()
				b.StartTimer()

				ctl.deferSync = true
				if tc.useBatch {
					if err := tr.SetBatch(batch); err != nil {
						b.Fatal(err)
					}
				} else {
					for _, kv := range batch {
						if err := tr.Set(kv.Key, kv.Val); err != nil {
							b.Fatal(err)
						}
					}
				}
				if _, err := tr.Snap(1); err != nil {
					b.Fatal(err)
				}
				if err := tr.Sync(); err != nil {
					b.Fatal(err)
				}
				ctl.deferSync = false
				if err := ctl.syncPending(); err != nil {
					b.Fatal(err)
				}

				b.StopTimer()
				_ = tr.Close()
				b.StartTimer()
			}
		})
	}
}

func copyTreeDir(srcDir, dstDir string) error {
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return err
	}
	for _, name := range []string{"mpt.tree1", "mpt.tree2", "mpt.disk"} {
		in, err := os.Open(filepath.Join(srcDir, name))
		if err != nil {
			return err
		}
		out, err := os.OpenFile(filepath.Join(dstDir, name), os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o666)
		if err != nil {
			in.Close()
			return err
		}
		if _, err := io.Copy(out, in); err != nil {
			in.Close()
			out.Close()
			return err
		}
		in.Close()
		if err := out.Sync(); err != nil {
			out.Close()
			return err
		}
		if err := out.Close(); err != nil {
			return err
		}
	}
	return nil
}

func measureTreeDepthStats(dt *diskTree) (leaves int, sumDepth int, maxDepth int) {
	root, err := dt.node(dt.hdr().root())
	if err != nil || root == nil {
		return 0, 0, 0
	}
	var walk func(n *diskNode, pbit, depth int)
	walk = func(n *diskNode, pbit, depth int) {
		if n.bit() <= pbit {
			leaves++
			sumDepth += depth
			if depth > maxDepth {
				maxDepth = depth
			}
			return
		}
		l, _ := dt.child(n.left())
		r, _ := dt.child(n.right())
		walk(l, n.bit(), depth+1)
		walk(r, n.bit(), depth+1)
	}
	walk(root, -1, 0)
	return leaves, sumDepth, maxDepth
}

func TestNVMeBenchMatrix(t *testing.T) {
	if os.Getenv("MPT_NVME_BENCH") == "" {
		t.Skip("set MPT_NVME_BENCH=1 (and MPT_BENCH_DIR=/path/on/disk if TMPDIR is tmpfs) to run disk benchmark matrix")
	}

	oldMax := pmem.SetMaxMem(16 << 30)
	defer pmem.SetMaxMem(oldMax)

	nvmeRoot := os.Getenv("MPT_BENCH_DIR")
	if nvmeRoot == "" {
		nvmeRoot = t.TempDir()
	}
	profDir := os.Getenv("MPT_PROF_DIR")

	runDir := filepath.Join(nvmeRoot, fmt.Sprintf("run-%d", time.Now().UnixNano()))
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(runDir)

	if profDir != "" {
		if err := os.MkdirAll(profDir, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	type variant struct {
		id       string
		label    string
		useBatch bool
		opts     []Option
	}
	variants := []variant{
		{"A", "Set loop (std I/O)", false, nil},
		{"B", "SetBatch (std I/O)", true, nil},
		{"C", "Set loop + Buf/Mmap", false, []Option{BufferLeafIO(true)}},
		{"D", "SetBatch + Buf/Mmap", true, []Option{BufferLeafIO(true)}},
	}

	type resultRow struct {
		scenario     string
		variantID    string
		variantLabel string
		keys         int
		mutateSync   time.Duration
		totalDur     time.Duration
		keysPerSec   float64
		allocBytesOp float64
		allocsOp     float64
		io           ioCounters
		rootHash     Hash
	}
	var allRows []resultRow

	runOne := func(scenario, profSlug string, dir string, create bool, ver int64, batch []KeyVal, v variant) resultRow {
		tr, ctl, err := openBenchDiskTree(dir, create, v.opts...)
		if err != nil {
			t.Fatalf("%s %s open: %v", scenario, v.id, err)
		}
		defer tr.Close()

		runtime.GC()
		ctl.resetCounters()

		var profFile *os.File
		if profDir != "" && profSlug != "" {
			pf, err := os.Create(filepath.Join(profDir, fmt.Sprintf("%s_%s.pprof", profSlug, v.id)))
			if err != nil {
				t.Fatal(err)
			}
			profFile = pf
			if err := pprof.StartCPUProfile(profFile); err != nil {
				t.Fatal(err)
			}
		}

		var msBefore, msAfter runtime.MemStats
		runtime.ReadMemStats(&msBefore)
		t0 := time.Now()

		ctl.deferSync = true
		if v.useBatch {
			if err := tr.SetBatch(batch); err != nil {
				t.Fatalf("%s %s SetBatch: %v", scenario, v.id, err)
			}
		} else {
			for _, kv := range batch {
				if err := tr.Set(kv.Key, kv.Val); err != nil {
					t.Fatalf("%s %s Set: %v", scenario, v.id, err)
				}
			}
		}
		snap, err := tr.Snap(ver)
		if err != nil {
			t.Fatalf("%s %s Snap: %v", scenario, v.id, err)
		}
		if err := tr.Sync(); err != nil {
			t.Fatalf("%s %s Sync: %v", scenario, v.id, err)
		}
		mutateSyncDur := time.Since(t0)

		ctl.deferSync = false
		if err := ctl.syncPending(); err != nil {
			t.Fatalf("%s %s syncPending: %v", scenario, v.id, err)
		}
		totalDur := time.Since(t0)
		runtime.ReadMemStats(&msAfter)

		if profFile != nil {
			pprof.StopCPUProfile()
			profFile.Close()
		}

		ioSnap := ctl.snapshotCounters()
		n := float64(len(batch))
		row := resultRow{
			scenario:     scenario,
			variantID:    v.id,
			variantLabel: v.label,
			keys:         len(batch),
			mutateSync:   mutateSyncDur,
			totalDur:     totalDur,
			keysPerSec:   n / totalDur.Seconds(),
			allocBytesOp: float64(msAfter.TotalAlloc-msBefore.TotalAlloc) / n,
			allocsOp:     float64(msAfter.Mallocs-msBefore.Mallocs) / n,
			io:           ioSnap,
			rootHash:     snap.Hash,
		}
		t.Logf("[%s | %s (%s)] keys=%d mutate+flush=%v total(w/fsync)=%v rate=%.0f keys/s | leafRead=%.3f/key (%.1fB) treeRead=%.3f/key | leafWrite=%.4f/key (%.1fB) treeWrite=%.4f/key (%.1fB) fsync=%d | %.1f B/op %.2f allocs/op | root=%x",
			scenario, v.id, v.label, len(batch),
			mutateSyncDur.Round(time.Millisecond), totalDur.Round(time.Millisecond), row.keysPerSec,
			float64(ioSnap.leafReadCalls)/n, float64(ioSnap.leafReadBytes)/n,
			float64(ioSnap.treeReadCalls)/n,
			float64(ioSnap.leafWriteCalls)/n, float64(ioSnap.leafWriteBytes)/n,
			float64(ioSnap.treeWriteCalls)/n, float64(ioSnap.treeWriteBytes)/n,
			ioSnap.fsyncCalls,
			row.allocBytesOp, row.allocsOp, snap.Hash[:8])
		return row
	}

	// 1. Cold Build N = 200,000
	cold200k := generateReversedDNSBatch(1001, 0, 200_000, 1)
	var sumKeyBytes int
	for _, kv := range cold200k {
		sumKeyBytes += len(kv.Key)
	}
	t.Logf("Fixture cold200k: %d keys, avg key=%.1f B, val=32 B", len(cold200k), float64(sumKeyBytes)/float64(len(cold200k)))

	var wantCold200k Hash
	for i, v := range variants {
		dir := filepath.Join(runDir, "cold200k-"+v.id)
		row := runOne("Cold 200k", "cold200k", dir, true, 1, cold200k, v)
		if i == 0 {
			wantCold200k = row.rootHash
		} else if row.rootHash != wantCold200k {
			t.Fatalf("Cold 200k %s root hash mismatch: got %v, want %v", v.id, row.rootHash, wantCold200k)
		}
		allRows = append(allRows, row)
		_ = os.RemoveAll(dir)
	}

	// 2. Cold Build N = 1,000,000 (also save one clean 1M base directory for Warm Incremental!)
	cold1M := generateReversedDNSBatch(2002, 0, 1_000_000, 1)
	base1MDir := filepath.Join(runDir, "base1M-template")
	var wantCold1M Hash
	for i, v := range variants {
		dir := filepath.Join(runDir, "cold1M-"+v.id)
		row := runOne("Cold 1M", "cold1m", dir, true, 1, cold1M, v)
		if i == 0 {
			wantCold1M = row.rootHash
		} else if row.rootHash != wantCold1M {
			t.Fatalf("Cold 1M %s root hash mismatch: got %v, want %v", v.id, row.rootHash, wantCold1M)
		}
		allRows = append(allRows, row)
		if v.id == "D" {
			// Measure exact tree depth on 1M tree before closing/copying, and preserve as warm base!
			trCheck, err := Open(filepath.Join(dir, "mpt.tree1"), filepath.Join(dir, "mpt.tree2"), filepath.Join(dir, "mpt.disk"))
			if err != nil {
				t.Fatal(err)
			}
			leaves, sumD, maxD := measureTreeDepthStats(trCheck.(*diskTree))
			t.Logf("1M-key Reversed-DNS Tree Topology: leaves=%d, avgDepth=%.2f, maxDepth=%d", leaves, float64(sumD)/float64(leaves), maxD)
			trCheck.Close()
			if err := os.Rename(dir, base1MDir); err != nil {
				t.Fatal(err)
			}
		} else {
			_ = os.RemoveAll(dir)
		}
	}

	// 3. Warm Incremental Batch: 200,000 sorted keys into 1,000,000-key tree (80% new = 160,000, 20% existing updates = 40,000)
	new160k := generateReversedDNSBatch(3003, 1_000_000, 160_000, 2)
	warmMap := make(map[string]Val, 200_000)
	for _, kv := range new160k {
		warmMap[string(kv.Key)] = kv.Val
	}
	// Pick 40,000 existing keys uniformly strided across cold1M with updated epoch-2 32B values.
	updValArena := make([]byte, 40_000*32)
	for i := range 40_000 {
		idx := (i * 25) % len(cold1M)
		k := cold1M[idx].Key
		var vb [16]byte
		binary.LittleEndian.PutUint64(vb[:8], uint64(idx))
		binary.LittleEndian.PutUint64(vb[8:], 2)
		sum := sha256.Sum256(vb[:])
		v := Val(updValArena[i*32 : (i+1)*32 : (i+1)*32])
		copy(v, sum[:])
		warmMap[string(k)] = v
	}
	warmBatch := make([]KeyVal, 0, len(warmMap))
	for ks, v := range warmMap {
		warmBatch = append(warmBatch, KeyVal{Key: Key(ks), Val: v})
	}
	slices.SortFunc(warmBatch, KeyVal.CompareKey)
	t.Logf("Fixture warm200k_on_1M: %d total sorted mutations (160,000 new inserts + 40,000 existing updates)", len(warmBatch))

	var wantWarm Hash
	for i, v := range variants {
		dir := filepath.Join(runDir, "warm-"+v.id)
		if err := copyTreeDir(base1MDir, dir); err != nil {
			t.Fatal(err)
		}
		row := runOne("Warm +200k (on 1M)", "warm200k", dir, false, 2, warmBatch, v)
		if i == 0 {
			wantWarm = row.rootHash
		} else if row.rootHash != wantWarm {
			t.Fatalf("Warm +200k %s root hash mismatch: got %v, want %v", v.id, row.rootHash, wantWarm)
		}
		allRows = append(allRows, row)
		_ = os.RemoveAll(dir)
	}
}
