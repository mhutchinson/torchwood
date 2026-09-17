// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pmem

import (
	"bytes"
	"io"
	"math/rand/v2"
	"testing"
)

// plainFile is a simple in-memory [File], with none of the crash-injection
// behaviour of testFile.
type plainFile struct {
	data []byte
}

func (f *plainFile) ReadAt(data []byte, off int64) (int, error) {
	if off < 0 || off >= int64(len(f.data)) {
		return 0, io.EOF
	}
	n := copy(data, f.data[off:])
	if n < len(data) {
		return n, io.ErrUnexpectedEOF
	}
	return n, nil
}

func (f *plainFile) WriteAt(data []byte, off int64) (int, error) {
	if off > int64(len(f.data)) {
		f.data = append(f.data, make([]byte, int(off)-len(f.data))...)
	}
	n := copy(f.data[off:], data)
	f.data = append(f.data, data[n:]...)
	return len(data), nil
}

func (f *plainFile) Sync() error { return nil }

// TestDiskBuffer checks that reads of the disk-only file agree with a shadow
// copy no matter how the pending append buffer happens to be positioned.
//
// Appends are coalesced in memory, so a read can be satisfied entirely from
// the file, entirely from the buffer, or from a file prefix followed by a
// buffer suffix. The last case is the interesting one: it only arises when a
// read starts before diskBufOff and continues past it.
func TestDiskBuffer(t *testing.T) {
	f1, f2, disk := new(plainFile), new(plainFile), new(plainFile)
	m, err := Create("magic", f1, f2, disk)
	check(t, err)

	// shadow is the expected logical content of the disk-only file.
	var shadow []byte
	r := rand.New(rand.NewPCG(1, 2))

	// readAll samples reads of many lengths at many offsets, so that they land
	// before, inside, and across the buffer boundary.
	readAll := func(what string) {
		t.Helper()
		if len(shadow) == 0 {
			return
		}
		for range 32 {
			off := int64(r.IntN(len(shadow)))
			n := 1 + r.IntN(len(shadow)-int(off))
			got := make([]byte, n)
			if err := m.ReadDisk(got, off); err != nil {
				t.Fatalf("%s: ReadDisk(len %d, off %d): %v", what, n, off, err)
			}
			if !bytes.Equal(got, shadow[off:off+int64(n)]) {
				t.Fatalf("%s: ReadDisk(len %d, off %d) = %x, want %x",
					what, n, off, got, shadow[off:off+int64(n)])
			}
		}
		// Whole-file read, which straddles whenever anything is buffered.
		got := make([]byte, len(shadow))
		check(t, m.ReadDisk(got, 0))
		if !bytes.Equal(got, shadow) {
			t.Fatalf("%s: full ReadDisk mismatch", what)
		}
	}

	// Append enough small records to fill and spill the buffer several times.
	for i := range 4000 {
		rec := make([]byte, 1+r.IntN(80))
		for j := range rec {
			rec[j] = byte(i)
		}
		check(t, m.WriteDisk(rec, m.DiskSize()))
		shadow = append(shadow, rec...)
		if m.DiskSize() != int64(len(shadow)) {
			t.Fatalf("DiskSize = %d, want %d", m.DiskSize(), len(shadow))
		}
		if i%37 == 0 {
			readAll("during appends")
		}
	}
	readAll("after appends")

	// Overwrite in place at random offsets. Some land inside the pending
	// buffer, some behind it in the file, exercising both paths.
	for range 100 {
		off := int64(r.IntN(len(shadow)))
		n := 1 + r.IntN(min(64, len(shadow)-int(off)))
		rec := make([]byte, n)
		for j := range rec {
			rec[j] = 0xA0 | byte(j&0xf)
		}
		check(t, m.WriteDisk(rec, off))
		copy(shadow[off:], rec)
		readAll("after overwrite")
	}

	// Interleave appends with overwrites, the pattern setVal produces when a
	// value grows and its leaf is reallocated at the end of the file.
	for range 100 {
		if r.IntN(2) == 0 {
			rec := []byte{1, 2, 3, 4, 5, 6, 7, 8}
			check(t, m.WriteDisk(rec, m.DiskSize()))
			shadow = append(shadow, rec...)
		} else {
			off := int64(r.IntN(len(shadow) - 4))
			rec := []byte{9, 9, 9, 9}
			check(t, m.WriteDisk(rec, off))
			copy(shadow[off:], rec)
		}
		readAll("interleaved")
	}

	// Flushing must not change what is readable.
	check(t, m.Sync())
	if len(m.diskBuf) != 0 {
		t.Fatalf("Sync left %d bytes buffered", len(m.diskBuf))
	}
	readAll("after Sync")

	// After Sync everything must also be in the file itself.
	raw := make([]byte, len(shadow))
	if _, err := disk.ReadAt(raw, m.diskOff); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, shadow) {
		t.Fatal("file content differs from logical content after Sync")
	}

	// And it must survive a reopen, which replays the patch log.
	check(t, m.Release())
	check(t, m.UnsafeUnmap())
	m2, err := Open("magic", f1, f2, disk)
	check(t, err)
	defer func() {
		check(t, m2.Release())
		check(t, m2.UnsafeUnmap())
	}()
	if m2.DiskSize() != int64(len(shadow)) {
		t.Fatalf("after reopen DiskSize = %d, want %d", m2.DiskSize(), len(shadow))
	}
	got := make([]byte, len(shadow))
	check(t, m2.ReadDisk(got, 0))
	if !bytes.Equal(got, shadow) {
		t.Fatal("content differs after reopen")
	}
}
