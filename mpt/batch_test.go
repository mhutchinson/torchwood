// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mpt

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
)

func checkDiskInvariants(t *testing.T, tr Tree) {
	t.Helper()
	dt, ok := tr.(*diskTree)
	if !ok {
		return
	}
	root, err := dt.node(dt.hdr().root())
	if err != nil {
		t.Fatalf("root node error: %v", err)
	}
	if root == nil {
		if dt.hdr().dirty() {
			t.Fatal("dirty header with nil root")
		}
		return
	}
	var sawNil bool
	h := verifyDiskNode(t, dt, root, -1, &sawNil)
	if !dt.hdr().dirty() && h != dt.hdr().hash() {
		t.Fatalf("header hash mismatch: got %v, want %v", dt.hdr().hash(), h)
	}
	if !sawNil {
		t.Fatal("lost unique nil (bit=-1) node in diskTree")
	}
}

func verifyDiskNode(t *testing.T, dt *diskTree, n *diskNode, pbit int, sawNil *bool) Hash {
	t.Helper()
	if n.bit() == -1 {
		if *sawNil {
			t.Fatal("multiple nil (bit=-1) nodes in diskTree")
		}
		*sawNil = true
	}
	if n.bit() <= pbit {
		k, v, err := n.keyVal(dt)
		if err != nil {
			t.Fatalf("leaf keyVal error: %v", err)
		}
		return hashLeaf(k, v)
	}
	ownKeyRaw, err := n.key(dt)
	if err != nil {
		t.Fatalf("inner node key error: %v", err)
	}
	ownKey := ownKeyRaw.Clone()
	leftLeaf, err := n.leftmost(dt, pbit)
	if err != nil {
		t.Fatalf("leftmost error: %v", err)
	}
	leftKey, err := leftLeaf.key(dt)
	if err != nil {
		t.Fatalf("leftKey error: %v", err)
	}
	if ownKey.overlap(leftKey) < n.bit() {
		t.Fatalf("inner node at bit %d holds leaf key %x outside its subtree prefix (leftmost %x)", n.bit(), ownKey, leftKey)
	}

	left, err := dt.child(n.left())
	if err != nil {
		t.Fatalf("left child error: %v", err)
	}
	right, err := dt.child(n.right())
	if err != nil {
		t.Fatalf("right child error: %v", err)
	}
	lh := verifyDiskNode(t, dt, left, n.bit(), sawNil)
	rh := verifyDiskNode(t, dt, right, n.bit(), sawNil)
	want := hashInner(n.bit(), lh, rh)
	if !n.dirty() && n.ihash() != want {
		t.Fatalf("inner node bit=%d hash mismatch: got %v, want %v", n.bit(), n.ihash(), want)
	}
	return want
}

func TestSetBatchValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		open func(t *testing.T) Tree
	}{
		{"mem", func(t *testing.T) Tree { return NewMemTree() }},
		{"disk", func(t *testing.T) Tree {
			dir := t.TempDir()
			tr, err := Create(filepath.Join(dir, "t1"), filepath.Join(dir, "t2"), filepath.Join(dir, "leaf"))
			check(t, err)
			return tr
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tr := tc.open(t)
			defer tr.Close()

			check(t, tr.Set(Key("m"), Val("init")))
			initSnap, err := tr.Snap(1)
			check(t, err)

			// Empty batch is a clean no-op: does not dirty tree or invalidate Path.
			check(t, tr.SetBatch(nil))
			check(t, tr.SetBatch([]KeyVal{}))
			if _, err := tr.Path(Key("m")); err != nil {
				t.Fatalf("Path after empty SetBatch failed: %v", err)
			}

			// Duplicate keys rejected without modifying tree.
			err = tr.SetBatch([]KeyVal{
				{Key: Key("a"), Val: Val("1")},
				{Key: Key("a"), Val: Val("2")},
			})
			if !errors.Is(err, ErrKeyOrder) {
				t.Fatalf("duplicate keys: got %v, want ErrKeyOrder", err)
			}

			// Unsorted keys rejected without modifying tree.
			err = tr.SetBatch([]KeyVal{
				{Key: Key("z"), Val: Val("1")},
				{Key: Key("b"), Val: Val("2")},
			})
			if !errors.Is(err, ErrKeyOrder) {
				t.Fatalf("unsorted keys: got %v, want ErrKeyOrder", err)
			}

			// Oversized key and val rejected without modifying tree.
			err = tr.SetBatch([]KeyVal{{Key: make(Key, MaxKeyLen+1), Val: Val("1")}})
			if !errors.Is(err, ErrKeySize) {
				t.Fatalf("oversized key: got %v, want ErrKeySize", err)
			}
			err = tr.SetBatch([]KeyVal{{Key: Key("ok"), Val: make(Val, MaxValLen+1)}})
			if !errors.Is(err, ErrValSize) {
				t.Fatalf("oversized val: got %v, want ErrValSize", err)
			}

			// Tree remains unmodified and clean after all rejected calls.
			if _, err := tr.Path(Key("m")); err != nil {
				t.Fatalf("Path after rejected SetBatch calls failed: %v", err)
			}
			afterSnap, err := tr.Snap(1)
			check(t, err)
			if afterSnap != initSnap {
				t.Fatalf("snapshot changed after rejected SetBatch: %v != %v", afterSnap, initSnap)
			}

			// Non-empty SetBatch marks tree dirty until Snap (Path returns ErrModifiedTree).
			check(t, tr.SetBatch([]KeyVal{{Key: Key("x"), Val: Val("y")}}))
			if _, err := tr.Path(Key("m")); !errors.Is(err, ErrModifiedTree) {
				t.Fatalf("Path before Snap: got %v, want ErrModifiedTree", err)
			}
			if _, err := tr.Predict([]KeyVal{{Key: Key("z"), Val: Val("w")}}); !errors.Is(err, ErrModifiedTree) {
				t.Fatalf("Predict before Snap: got %v, want ErrModifiedTree", err)
			}
		})
	}
}

func TestSetBatchVariableLengthAndPaddingEdges(t *testing.T) {
	k1023 := bytes.Repeat([]byte{'a'}, MaxKeyLen-1)
	k1024Zero := append(bytes.Clone(k1023), 0x00)
	k1024FF := append(bytes.Clone(k1023), 0xFF)

	edgeKeys := []Key{
		Key(""),
		Key("\x00"),
		Key("\x00\x00"),
		Key("\x00\xff"),
		Key("\x00\xff\x00"),
		Key("\x00\xff\xff"),
		Key("abc"),
		Key("abc\x00"),
		Key("abc\x00\x00"),
		Key("abc\x00\xfe"),
		Key("abc\x00\xff"),
		Key("abc\x00\xff\x00"),
		Key("abc\x00\xff\xff"),
		Key("abd"),
		k1023,
		k1024Zero,
		k1024FF,
	}

	var batch []KeyVal
	for i, k := range edgeKeys {
		batch = append(batch, KeyVal{Key: k.Clone(), Val: fmt.Appendf(nil, "val-%d", i)})
	}
	slices.SortFunc(batch, KeyVal.CompareKey)

	dir := t.TempDir()
	ref := newDiskTree()
	defer ref.Close()
	got, err := Create(filepath.Join(dir, "t1"), filepath.Join(dir, "t2"), filepath.Join(dir, "leaf"))
	check(t, err)
	defer got.Close()

	for _, kv := range batch {
		check(t, ref.Set(kv.Key, kv.Val))
	}
	check(t, got.SetBatch(batch))

	sRef, err := ref.Snap(1)
	check(t, err)
	sGot, err := got.Snap(1)
	check(t, err)
	if sGot != sRef {
		t.Fatalf("padding edge snapshot mismatch: got %v, want %v", sGot, sRef)
	}
	checkDiskInvariants(t, got)

	// Update threaded inner-node leaves with larger (>256B) values forcing reallocation.
	var batch2 []KeyVal
	for i, kv := range batch {
		if i%2 == 0 {
			batch2 = append(batch2, KeyVal{
				Key: kv.Key.Clone(),
				Val: bytes.Repeat([]byte{byte('A' + i)}, 300),
			})
		}
	}
	batch2 = append(batch2,
		KeyVal{Key: Key("abc\x00\x7f"), Val: bytes.Repeat([]byte{'Z'}, 400)},
		KeyVal{Key: Key(" middle"), Val: Val("short")},
	)
	slices.SortFunc(batch2, KeyVal.CompareKey)

	for _, kv := range batch2 {
		check(t, ref.Set(kv.Key, kv.Val))
	}
	check(t, got.SetBatch(batch2))

	sRef2, err := ref.Snap(2)
	check(t, err)
	sGot2, err := got.Snap(2)
	check(t, err)
	if sGot2 != sRef2 {
		t.Fatalf("batch2 snapshot mismatch: got %v, want %v", sGot2, sRef2)
	}
	checkDiskInvariants(t, got)

	for _, kv := range batch {
		pRef, err := ref.Path(kv.Key)
		check(t, err)
		pGot, err := got.Path(kv.Key)
		check(t, err)
		if !bytes.Equal(pRef, pGot) {
			t.Fatalf("Path(%x) mismatch:\ngot  %x\nwant %x", kv.Key, pGot, pRef)
		}
	}
}

func TestSetBatchDifferentialRandom(t *testing.T) {
	r := rand.New(rand.NewPCG(20261124, 99))

	genKey := func(id int) Key {
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], uint64(id))
		h := sha256.Sum256(b[:])
		switch id % 4 {
		case 0:
			return fmt.Appendf(nil, "uk.co.domain-%04d.sub-%x", id%64, h[:6])
		case 1:
			return fmt.Appendf(nil, "com.example.host-%03d.%x", id%32, h[:10])
		case 2:
			return Key(bytes.Clone(h[:1+id%32]))
		default:
			return Key(bytes.Clone(h[:]))
		}
	}

	genVal := func(round, id int) Val {
		n := 16 + (round*37+id*13)%180
		out := make(Val, n)
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], uint64(round)<<32|uint64(id))
		h := sha256.Sum256(b[:])
		for i := range out {
			out[i] = h[i%32] ^ byte(i)
		}
		return out
	}

	for trial := range 8 {
		t.Run(fmt.Sprintf("trial_%d", trial), func(t *testing.T) {
			dir := t.TempDir()

			refMem := NewMemTree()
			defer refMem.Close()
			refDisk := newDiskTree()
			defer refDisk.Close()

			batchMem := NewMemTree()
			defer batchMem.Close()
			batchDisk, err := Create(filepath.Join(dir, "t1"), filepath.Join(dir, "t2"), filepath.Join(dir, "leaf"))
			check(t, err)
			defer func() {
				if batchDisk != nil {
					batchDisk.Close()
				}
			}()

			var activeKeys []Key
			seen := make(map[string]bool)

			batchSizes := []int{1, 1, 5, 25, 120, 400, 80, 1, 250, 60, 150, 50}
			var ver int64

			for round, bsize := range batchSizes {
				if round%4 == 2 {
					soloK := genKey(1_000_000 + trial*100 + round)
					soloV := genVal(round, 999999)
					if !seen[string(soloK)] {
						seen[string(soloK)] = true
						activeKeys = append(activeKeys, soloK)
					}
					for _, tr := range []Tree{refMem, refDisk, batchMem, batchDisk} {
						check(t, tr.Set(soloK, soloV))
					}
				}

				m := make(map[string]Val, bsize)
				for len(m) < bsize {
					var k Key
					if len(activeKeys) > 0 && r.IntN(100) < 35 {
						k = activeKeys[r.IntN(len(activeKeys))]
					} else {
						k = genKey(r.IntN(50000) + round*10000 + trial*100000)
						if !seen[string(k)] {
							seen[string(k)] = true
							activeKeys = append(activeKeys, k)
						}
					}
					m[string(k)] = genVal(round, len(m))
				}

				changes := make([]KeyVal, 0, len(m))
				for ks, val := range m {
					changes = append(changes, KeyVal{Key: Key(ks), Val: val})
				}
				slices.SortFunc(changes, KeyVal.CompareKey)

				var predicted Hash
				havePredict := round%4 != 2
				if havePredict {
					var err error
					predicted, err = batchDisk.Predict(changes)
					check(t, err)
				}

				for _, kv := range changes {
					check(t, refMem.Set(kv.Key, kv.Val))
					check(t, refDisk.Set(kv.Key, kv.Val))
				}
				check(t, batchMem.SetBatch(changes))
				check(t, batchDisk.SetBatch(changes))

				if round == 5 {
					extra := []KeyVal{
						{Key: Key("00-back-to-back"), Val: Val("v1")},
						{Key: Key("zz-back-to-back"), Val: Val("v2")},
					}
					for _, kv := range extra {
						if !seen[string(kv.Key)] {
							seen[string(kv.Key)] = true
							activeKeys = append(activeKeys, kv.Key)
						}
						check(t, refMem.Set(kv.Key, kv.Val))
						check(t, refDisk.Set(kv.Key, kv.Val))
					}
					check(t, batchMem.SetBatch(extra))
					check(t, batchDisk.SetBatch(extra))
					havePredict = false
				}

				ver++
				wantSnap, err := refMem.Snap(ver)
				check(t, err)
				if havePredict && predicted != wantSnap.Hash {
					t.Fatalf("round %d: Predict=%v != Snap=%v", round, predicted, wantSnap.Hash)
				}

				for name, tr := range map[string]Tree{
					"refDisk":   refDisk,
					"batchMem":  batchMem,
					"batchDisk": batchDisk,
				} {
					gotSnap, err := tr.Snap(ver)
					check(t, err)
					if gotSnap != wantSnap {
						t.Fatalf("round %d (%s): Snap = %v, want %v", round, name, gotSnap, wantSnap)
					}
					checkDiskInvariants(t, tr)
				}
			}

			check(t, batchDisk.Sync())
			check(t, batchDisk.Close())
			batchDisk = nil

			reopenDisk, err := Open(filepath.Join(dir, "t1"), filepath.Join(dir, "t2"), filepath.Join(dir, "leaf"))
			check(t, err)
			defer reopenDisk.Close()

			wantSnap, err := refDisk.Snap(-1)
			check(t, err)

			for name, tr := range map[string]Tree{
				"reopenDisk": reopenDisk,
				"batchMem":   batchMem,
			} {
				gotVer, gotExact := tr.Version()
				if gotVer != ver || !gotExact {
					t.Fatalf("%s Version() = (%d, %v), want (%d, true)", name, gotVer, gotExact, ver)
				}
				gotSnap, err := tr.Snap(-1)
				check(t, err)
				if gotSnap != wantSnap {
					t.Fatalf("%s Snap(-1) = %v, want %v", name, gotSnap, wantSnap)
				}
				checkDiskInvariants(t, tr)

				var wantAll, gotAll []KeyVal
				for kv, err := range refDisk.Scan(nil) {
					check(t, err)
					wantAll = append(wantAll, KeyVal{kv.Key.Clone(), kv.Val.Clone()})
				}
				for kv, err := range tr.Scan(nil) {
					check(t, err)
					gotAll = append(gotAll, KeyVal{kv.Key.Clone(), kv.Val.Clone()})
				}
				if !slices.EqualFunc(wantAll, gotAll, KeyVal.Equal) {
					t.Fatalf("%s full Scan mismatch (len %d vs %d)", name, len(gotAll), len(wantAll))
				}

				for _, k := range activeKeys {
					pWant, err := refDisk.Path(k)
					check(t, err)
					pGot, err := tr.Path(k)
					check(t, err)
					if !bytes.Equal(pWant, pGot) {
						t.Fatalf("%s Path(%x) differs from refDisk", name, k)
					}
					val, ok, proof, err := ProveLookup(k, pGot)
					check(t, err)
					if !ok {
						t.Fatalf("%s key %x missing", name, k)
					}
					check(t, VerifyLookup(wantSnap, k, val, true, proof))
				}

				for i := range 100 {
					miss := genKey(9_000_000 + i)
					pWant, err := refDisk.Path(miss)
					check(t, err)
					pGot, err := tr.Path(miss)
					check(t, err)
					if !bytes.Equal(pWant, pGot) {
						t.Fatalf("%s absent Path(%x) differs from refDisk", name, miss)
					}
					val, ok, proof, err := ProveLookup(miss, pGot)
					check(t, err)
					if ok {
						t.Fatalf("%s absent key %x unexpectedly found", name, miss)
					}
					check(t, VerifyLookup(wantSnap, miss, val, false, proof))
				}

				for _, pfx := range [][]byte{
					[]byte("uk.co."),
					[]byte("uk.co.domain-0001."),
					[]byte("com.example."),
					[]byte("nonexistent."),
					{0x00},
					{0xff},
				} {
					pWant, err := refDisk.Path(pfx)
					check(t, err)
					pGot, err := tr.Path(pfx)
					check(t, err)
					if !bytes.Equal(pWant, pGot) {
						t.Fatalf("%s prefix Path(%x) differs from refDisk", name, pfx)
					}
					subHash, proof, err := ProvePrefix(pfx, pGot)
					check(t, err)
					check(t, VerifyPrefix(wantSnap, pfx, subHash, proof))
				}
			}
		})
	}
}

func TestSetBatchCrashRecovery(t *testing.T) {
	t.Run("cold_empty_tree", func(t *testing.T) {
		testSetBatchCrashRecovery(t, false)
	})
	t.Run("warm_non_empty_tree", func(t *testing.T) {
		testSetBatchCrashRecovery(t, true)
	})
}

func testSetBatchCrashRecovery(t *testing.T, seedFirst bool) {
	var files [3]memFile
	xtree, err := New(&files[0], &files[1], &files[2])
	check(t, err)
	dt := xtree.(*diskTree)

	targetVer := int64(1)
	if seedFirst {
		var v1Batch []KeyVal
		for i := range 20 {
			v1Batch = append(v1Batch, KeyVal{Key: Key(v(i * 3)), Val: v(i)})
		}
		slices.SortFunc(v1Batch, KeyVal.CompareKey)
		check(t, dt.SetBatch(v1Batch))
		_, err = dt.Snap(1)
		check(t, err)
		check(t, dt.Sync())
		targetVer = 2
	} else {
		_, err = dt.Snap(0)
		check(t, err)
		check(t, dt.Sync())
	}

	var batch []KeyVal
	for i := range 25 {
		batch = append(batch, KeyVal{Key: Key(v(i * 2)), Val: v(1000 + i)})
	}
	slices.SortFunc(batch, KeyVal.CompareKey)

	wantHash, err := dt.Predict(batch)
	check(t, err)
	check(t, dt.Close())

	type diskSnap struct {
		f0, f1, f2 []byte
	}
	var states []diskSnap
	var hfiles [3]*hookFile
	snapshotAll := func() {
		states = append(states, diskSnap{
			f0: bytes.Clone(hfiles[0].data),
			f1: bytes.Clone(hfiles[1].data),
			f2: bytes.Clone(hfiles[2].data),
		})
	}
	for i := range 3 {
		hfiles[i] = &hookFile{memFile: memFile{data: bytes.Clone(files[i].data)}}
	}
	for i := range 3 {
		hfiles[i].onWrite = snapshotAll
	}

	xtree2, err := New(hfiles[0], hfiles[1], hfiles[2])
	check(t, err)
	dt2 := xtree2.(*diskTree)
	defer dt2.Close()
	dt2.pmem.SetConstantFlushing(true)

	check(t, dt2.SetBatch(batch))
	_, err = dt2.Snap(targetVer)
	check(t, err)
	check(t, dt2.Sync())

	if len(states) < 10 {
		t.Fatalf("expected many intermediate states, got %d", len(states))
	}

	stride := max(1, len(states)/40)
	for idx := 0; idx < len(states); idx += stride {
		st := states[idx]

		// Verify Snap(-1) directly on the crashed state never panics (e.g. on empty root).
		{
			f0 := &memFile{data: bytes.Clone(st.f0)}
			f1 := &memFile{data: bytes.Clone(st.f1)}
			f2 := &memFile{data: bytes.Clone(st.f2)}
			rec, err := New(f0, f1, f2)
			if err != nil {
				t.Fatalf("state %d/%d reopen failed: %v", idx, len(states), err)
			}
			checkDiskInvariants(t, rec)
			_, err = rec.Snap(-1)
			check(t, err)
			checkDiskInvariants(t, rec)
			check(t, rec.Close())
		}

		// Verify standard replay recovery via both Set and SetBatch.
		for _, useBatchReplay := range []bool{false, true} {
			f0 := &memFile{data: bytes.Clone(st.f0)}
			f1 := &memFile{data: bytes.Clone(st.f1)}
			f2 := &memFile{data: bytes.Clone(st.f2)}

			rec, err := New(f0, f1, f2)
			if err != nil {
				t.Fatalf("state %d/%d reopen failed: %v", idx, len(states), err)
			}
			checkDiskInvariants(t, rec)

			ver, exact := rec.Version()
			if ver < targetVer-1 {
				rec.Close()
				t.Fatalf("state %d: version=%d < %d", idx, ver, targetVer-1)
			}
			if ver == targetVer-1 && !exact {
				if useBatchReplay {
					check(t, rec.SetBatch(batch))
				} else {
					for _, kv := range batch {
						check(t, rec.Set(kv.Key, kv.Val))
					}
				}
				snap, err := rec.Snap(targetVer)
				check(t, err)
				if snap.Hash != wantHash {
					rec.Close()
					t.Fatalf("state %d (batchReplay=%v): recovered hash=%v, want %v", idx, useBatchReplay, snap.Hash, wantHash)
				}
			} else if ver == targetVer {
				snap, err := rec.Snap(-1)
				check(t, err)
				if snap.Hash != wantHash {
					rec.Close()
					t.Fatalf("state %d: ver=%d hash=%v, want %v", idx, targetVer, snap.Hash, wantHash)
				}
			}
			checkDiskInvariants(t, rec)
			check(t, rec.Close())
		}
	}
}

type hookFile struct {
	memFile
	onWrite func()
}

func (h *hookFile) WriteAt(b []byte, off int64) (int, error) {
	n, err := h.memFile.WriteAt(b, off)
	if h.onWrite != nil {
		h.onWrite()
	}
	return n, err
}

func TestSetBatchConcurrentReadersRace(t *testing.T) {
	dir := t.TempDir()
	tr, err := Create(
		filepath.Join(dir, "tree1"),
		filepath.Join(dir, "tree2"),
		filepath.Join(dir, "disk"),
	)
	check(t, err)
	defer func() {
		if tr != nil {
			tr.Close()
		}
	}()

	batch := generateBenchBatch(777, 0, 1500, 1)
	check(t, tr.SetBatch(batch))
	snap, err := tr.Snap(1)
	check(t, err)
	check(t, tr.Sync())

	predChanges := generateBenchBatch(888, 1500, 32, 2)
	wantPred, err := tr.Predict(predChanges)
	check(t, err)

	var wg sync.WaitGroup
	for g := range 16 {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for iter := range 30 {
				kv := batch[(worker*31+iter)%len(batch)]
				path, err := tr.Path(kv.Key)
				if err != nil {
					t.Errorf("Path(%x): %v", kv.Key, err)
					return
				}
				val, ok, proof, err := ProveLookup(kv.Key, path)
				if err != nil || !ok || !bytes.Equal(val, kv.Val) {
					t.Errorf("ProveLookup(%x) = (%x, %v, %v)", kv.Key, val, ok, err)
					return
				}
				if err := VerifyLookup(snap, kv.Key, val, true, proof); err != nil {
					t.Errorf("VerifyLookup(%x): %v", kv.Key, err)
					return
				}

				miss := Key(fmt.Sprintf("com.nonexistent.%d.%d", worker, iter))
				mPath, err := tr.Path(miss)
				if err != nil {
					t.Errorf("Path(miss): %v", err)
					return
				}
				mVal, ok, mProof, err := ProveLookup(miss, mPath)
				if err != nil || ok {
					t.Errorf("ProveLookup(miss) = (%v, %v)", ok, err)
					return
				}
				if err := VerifyLookup(snap, miss, mVal, false, mProof); err != nil {
					t.Errorf("VerifyLookup(miss): %v", err)
					return
				}

				pref := kv.Key[:min(len(kv.Key), 10)]
				pPath, err := tr.Path(pref)
				if err != nil {
					t.Errorf("Path(%q): %v", pref, err)
					return
				}
				subHash, pProof, err := ProvePrefix(pref, pPath)
				if err != nil {
					t.Errorf("ProvePrefix(%q): %v", pref, err)
					return
				}
				if err := VerifyPrefix(snap, pref, subHash, pProof); err != nil {
					t.Errorf("VerifyPrefix(%q): %v", pref, err)
					return
				}
				for item, err := range tr.Scan(pref) {
					if err != nil {
						t.Errorf("Scan(%q): %v", pref, err)
						return
					}
					if !item.Key.HasPrefix(pref) {
						t.Errorf("Scan(%q) yielded %q", pref, item.Key)
						return
					}
				}

				gotPred, err := tr.Predict(predChanges)
				if err != nil || gotPred != wantPred {
					t.Errorf("Predict = (%v, %v), want %v", gotPred, err, wantPred)
					return
				}
			}
		}(g)
	}
	wg.Wait()

	check(t, tr.Close())
	tr = nil
}

func generateBenchBatch(seed uint64, startID, count int, valEpoch uint64) []KeyVal {
	tlds := []string{"com", "com", "net", "org", "io", "dev", "de", "uk.co"}
	subs := []string{"www", "api", "cdn", "mail", "auth", "app", "edge", "portal"}
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	kvs := make([]KeyVal, count)
	valArena := make([]byte, count*32)
	keyArena := make([]byte, 0, count*32)
	numDomains := max(256, (startID+count)/8)

	for i := range count {
		id := startID + i
		startOff := len(keyArena)
		keyArena = fmt.Appendf(keyArena, "%s.dom%05x.%s-%06x",
			tlds[id%len(tlds)], r.IntN(numDomains), subs[(id/3)%len(subs)], id)
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

func copyBenchTree(srcDir, dstDir string) error {
	for _, name := range []string{"t1", "t2", "leaf"} {
		in, err := os.Open(filepath.Join(srcDir, name))
		if err != nil {
			return err
		}
		out, err := os.OpenFile(filepath.Join(dstDir, name), os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o666)
		if err != nil {
			in.Close()
			return err
		}
		_, err = io.Copy(out, in)
		in.Close()
		if err != nil {
			out.Close()
			return err
		}
		if err := out.Close(); err != nil {
			return err
		}
	}
	return nil
}

func BenchmarkSetBatch(b *testing.B) {
	cold10k := generateBenchBatch(42, 0, 10_000, 1)
	warm2k := generateBenchBatch(99, 10_000, 1_600, 2)
	for i := range 400 {
		kv := cold10k[i*25]
		warm2k = append(warm2k, KeyVal{Key: kv.Key, Val: bytes.Repeat([]byte{byte(i)}, 32)})
	}
	slices.SortFunc(warm2k, KeyVal.CompareKey)

	baseDir := b.TempDir()
	baseTr, err := Create(filepath.Join(baseDir, "t1"), filepath.Join(baseDir, "t2"), filepath.Join(baseDir, "leaf"))
	if err != nil {
		b.Fatal(err)
	}
	if err := baseTr.SetBatch(cold10k); err != nil {
		b.Fatal(err)
	}
	if _, err := baseTr.Snap(1); err != nil {
		b.Fatal(err)
	}
	if err := baseTr.Close(); err != nil {
		b.Fatal(err)
	}

	modes := []struct {
		name     string
		lazy     bool
		useBatch bool
	}{
		{"Set_Eager", false, false},
		{"Set_LazyHash", true, false},
		{"SetBatch", false, true},
	}

	for _, sc := range []struct {
		name  string
		warm  bool
		batch []KeyVal
	}{
		{"Cold_10K", false, cold10k},
		{"Warm_2K_on_10K", true, warm2k},
	} {
		for _, m := range modes {
			b.Run(sc.name+"/"+m.name, func(b *testing.B) {
				prevLazy := lazyHash
				lazyHash = m.lazy
				defer func() { lazyHash = prevLazy }()

				b.ReportAllocs()
				for b.Loop() {
					b.StopTimer()
					dir := b.TempDir()
					var tr Tree
					var err error
					if sc.warm {
						if err := copyBenchTree(baseDir, dir); err != nil {
							b.Fatal(err)
						}
						tr, err = Open(filepath.Join(dir, "t1"), filepath.Join(dir, "t2"), filepath.Join(dir, "leaf"))
					} else {
						tr, err = Create(filepath.Join(dir, "t1"), filepath.Join(dir, "t2"), filepath.Join(dir, "leaf"))
					}
					if err != nil {
						b.Fatal(err)
					}
					b.StartTimer()

					if m.useBatch {
						if err := tr.SetBatch(sc.batch); err != nil {
							b.Fatal(err)
						}
					} else {
						for _, kv := range sc.batch {
							if err := tr.Set(kv.Key, kv.Val); err != nil {
								b.Fatal(err)
							}
						}
					}
					if _, err := tr.Snap(2); err != nil {
						b.Fatal(err)
					}

					b.StopTimer()
					_ = tr.Close()
					b.StartTimer()
				}
			})
		}
	}
}
