// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pmem

import (
	"bytes"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

func TestBufferedDiskAndMmap(t *testing.T) {
	dir := t.TempDir()
	openFiles := func(flag int) (*os.File, *os.File, *os.File) {
		t.Helper()
		f1, err := os.OpenFile(filepath.Join(dir, "tree1"), flag, 0o666)
		check(t, err)
		f2, err := os.OpenFile(filepath.Join(dir, "tree2"), flag, 0o666)
		check(t, err)
		disk, err := os.OpenFile(filepath.Join(dir, "disk"), flag, 0o666)
		check(t, err)
		return f1, f2, disk
	}

	f1, f2, disk := openFiles(os.O_RDWR | os.O_CREATE | os.O_EXCL)
	m, err := Create("magic", f1, f2, disk)
	check(t, err)
	m.EnableBufferedDisk()
	if m.diskMap == nil {
		t.Fatal("expected diskMap to be initialized for *os.File")
	}

	var shadow []byte
	r := rand.New(rand.NewPCG(11, 22))

	verifyReads := func(stage string) {
		t.Helper()
		if len(shadow) == 0 {
			return
		}
		for range 32 {
			off := int64(r.IntN(len(shadow)))
			n := 1 + r.IntN(len(shadow)-int(off))
			got := make([]byte, n)
			if err := m.ReadDisk(got, off); err != nil {
				t.Fatalf("%s: ReadDisk(off=%d, len=%d): %v", stage, off, n, err)
			}
			if !bytes.Equal(got, shadow[off:off+int64(n)]) {
				t.Fatalf("%s: ReadDisk mismatch at off=%d len=%d", stage, off, n)
			}
		}
		// Full-range read straddling flushed mmap region and pending diskBuf.
		got := make([]byte, len(shadow))
		check(t, m.ReadDisk(got, 0))
		if !bytes.Equal(got, shadow) {
			t.Fatalf("%s: full ReadDisk mismatch", stage)
		}
	}

	// Spill the 64 KiB buffer multiple times.
	for i := range 4000 {
		rec := make([]byte, 1+r.IntN(90))
		for j := range rec {
			rec[j] = byte(i + j)
		}
		check(t, m.WriteDisk(rec, m.DiskSize()))
		shadow = append(shadow, rec...)
		if i%97 == 0 {
			verifyReads("during appends")
		}
	}
	verifyReads("after appends")

	// Overwrite both flushed (mmap) regions and un-flushed tail buffer regions.
	for range 120 {
		off := int64(r.IntN(len(shadow) - 16))
		rec := make([]byte, 1+r.IntN(16))
		for j := range rec {
			rec[j] = 0xC0 ^ byte(j)
		}
		check(t, m.WriteDisk(rec, off))
		copy(shadow[off:], rec)
		verifyReads("after overwrite")
	}

	check(t, m.Sync())
	verifyReads("after Sync")
	check(t, m.Release())
	check(t, m.UnsafeUnmap())
	_ = f1.Close()
	_ = f2.Close()
	_ = disk.Close()

	// Reopen with buffered/mmap enabled and verify full persistence.
	f1, f2, disk = openFiles(os.O_RDWR)
	defer f1.Close()
	defer f2.Close()
	defer disk.Close()

	m2, err := Open("magic", f1, f2, disk)
	check(t, err)
	m2.EnableBufferedDisk()
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
		t.Fatal("content mismatch after reopen")
	}
}
