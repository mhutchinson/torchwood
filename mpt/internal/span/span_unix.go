//go:build !plan9 && !windows

// Package span implements growable memory spans.
package span

import (
	"fmt"

	"golang.org/x/sys/unix"
)

type Span struct {
	max      int
	alloc    int
	mem      []byte
	released []byte
}

const pageSize = 4 << 20

func round(n int) int {
	return (n + pageSize - 1) &^ (pageSize - 1)
}

// Reserve returns a Span with zero memory footprint
// but with space reserved to expand to at most max bytes.
func Reserve(max int) (*Span, error) {
	r := round(max)
	mem, err := unix.Mmap(-1, 0, r, unix.PROT_NONE, unix.MAP_ANON|unix.MAP_PRIVATE|unix.MAP_NORESERVE)
	if err != nil {
		return nil, fmt.Errorf("span.Reserve %d: %w", r, err)
	}
	return &Span{max: max, mem: mem}, nil
}

// Expand expands the accessible memory to at least n bytes
// and returns a slice of length n and capacity n.
// Calling Expand with a small n does not release memory
// from an earlier call with a larger n.
func (s *Span) Expand(n int) ([]byte, error) {
	r := round(n)
	if s.alloc < r {
		err := unix.Mprotect(s.mem[s.alloc:r], unix.PROT_READ|unix.PROT_WRITE)
		if err != nil {
			return nil, fmt.Errorf("span.Expand %d..%d: %w", s.alloc, r, err)
		}
		s.alloc = r
	}
	return s.mem[:n:n], nil
}

// Release releases the memory for the span.
// If previously returned memory is accessed after calling Release,
// the accesses will fault, which will crash the program
// or else panic, depending on the use of [runtime/debug.SetPanicOnFault].
// [Span.UnsafeUnmap] releases the virtual memory for the span,
// but it is unsafe and rarely necessary to use.
func (s *Span) Release() error {
	if s.mem == nil {
		return fmt.Errorf("span.Release already called")
	}
	if s.alloc > 0 {
		if err := unix.Mprotect(s.mem[:s.alloc], unix.PROT_NONE); err != nil {
			return fmt.Errorf("span.Release: mprotect: %w", err)
		}
		err := unix.Madvise(s.mem[:s.alloc], unix.MADV_FREE)
		if err == unix.EINVAL {
			// MADV_FREE is missing before Linux 4.5 and in gVisor.
			err = unix.Madvise(s.mem[:s.alloc], unix.MADV_DONTNEED)
		}
		if err != nil {
			return fmt.Errorf("span.Release: madvise: %w", err)
		}
	}
	s.released = s.mem
	s.mem = nil
	return nil
}

// UnsafeUnmap releases the virtual memory for the span,
// making accesses to the previously returned memory behave unpredictably.
// Perhaps they will still fault, but if the operating system reuses the
// virtual address space, they might instead access unrelated memory.
// On 64-bit systems, the virtual address space available to processes
// is typically on the order of 2⁶³ bytes.
// Unless [Reserve] is being called for sizes totaling beyond that amount,
// programs can use [Span.Release] without UnsafeUnmap and avoid
// the unsafe behavior.
//
// If [Span.Release] has not been called, UnsafeUnmap does nothing
// but return an error.
func (s *Span) UnsafeUnmap() error {
	if s.mem != nil {
		return fmt.Errorf("Span.UnsafeUnmap without Span.Release")
	}
	if s.released == nil {
		return fmt.Errorf("Span.UnsafeUnmap already called")
	}
	if err := unix.Munmap(s.released); err != nil {
		return fmt.Errorf("span.Release: %w", err)
	}
	s.released = nil
	return nil
}

// MapFile memory-maps regular file fd up to maxLen bytes with MAP_SHARED.
// It returns the mapped slice, whether the mapping is writable, the current
// file size on disk, and whether mapping succeeded.
func MapFile(fd uintptr, maxLen int) (data []byte, writable bool, fileSize int64, ok bool) {
	var st unix.Stat_t
	if err := unix.Fstat(int(fd), &st); err != nil || (st.Mode&unix.S_IFMT) != unix.S_IFREG {
		return nil, false, 0, false
	}
	if mapped, err := unix.Mmap(int(fd), 0, maxLen, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED); err == nil {
		return mapped, true, int64(st.Size), true
	}
	if mapped, err := unix.Mmap(int(fd), 0, maxLen, unix.PROT_READ, unix.MAP_SHARED); err == nil {
		return mapped, false, int64(st.Size), true
	}
	return nil, false, 0, false
}

// SyncFile flushes dirty mapped pages in data[:length] to disk.
func SyncFile(data []byte, length int) error {
	if length <= 0 {
		return nil
	}
	return unix.Msync(data[:min(length, len(data))], unix.MS_SYNC)
}

// UnmapFile unmaps a slice previously returned by [MapFile].
func UnmapFile(data []byte) error {
	return unix.Munmap(data)
}
