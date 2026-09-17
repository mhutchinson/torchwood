// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

// Mptload loads a list of keys into a database.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"filippo.io/torchwood/mpt"
)

var (
	countFlag = flag.Int("n", 0, "generate N keys in-flight instead of reading file")
	stepFlag  = flag.Int("step", 1000000, "log interval step")
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: mptload [flags] db1 db2 leafdb [keys.txt]\n")
	flag.PrintDefaults()
	os.Exit(2)
}

func main() {
	log.SetPrefix("mptload: ")
	flag.Usage = usage
	flag.Parse()

	if *countFlag > 0 && flag.NArg() != 3 {
		usage()
	}
	if *countFlag == 0 && flag.NArg() != 4 {
		usage()
	}

	file1, file2, file3 := flag.Arg(0), flag.Arg(1), flag.Arg(2)
	tree, err := mpt.Create(file1, file2, file3)
	if err != nil {
		log.Fatal(err)
	}

	start := time.Now()
	lastStep := start
	totalKeys := *countFlag
	step := *stepFlag
	if step <= 0 {
		step = 1000000
	}

	if totalKeys > 0 {
		var buf [8]byte
		for i := 1; i <= totalKeys; i++ {
			binary.BigEndian.PutUint64(buf[:], uint64(i))
			h := sha256.Sum256(buf[:])
			if err := tree.Set(h[:], h[:]); err != nil {
				log.Fatal(err)
			}
			if i%step == 0 {
				now := time.Now()
				stepElapsed := now.Sub(lastStep)
				totalElapsed := now.Sub(start)
				intervalRate := float64(step) / stepElapsed.Seconds()
				overallRate := float64(i) / totalElapsed.Seconds()
				log.Printf("stored %d / %d (interval: %.0f ops/s in %s, overall: %.0f ops/s, elapsed %s)",
					i, totalKeys, intervalRate, stepElapsed.Round(time.Millisecond),
					overallRate, totalElapsed.Round(time.Millisecond))
				lastStep = now
			}
		}
	} else {
		keysFile := flag.Arg(3)
		data, err := os.ReadFile(keysFile)
		if err != nil {
			log.Fatal(err)
		}
		n := 0
		for line := range bytes.Lines(data) {
			h := sha256.Sum256(line)
			if err := tree.Set(h[:], h[:]); err != nil {
				log.Fatal(err)
			}
			n++
			if n%step == 0 {
				now := time.Now()
				stepElapsed := now.Sub(lastStep)
				totalElapsed := now.Sub(start)
				intervalRate := float64(step) / stepElapsed.Seconds()
				overallRate := float64(n) / totalElapsed.Seconds()
				log.Printf("stored %d (interval: %.0f ops/s in %s, overall: %.0f ops/s, elapsed %s)",
					n, intervalRate, stepElapsed.Round(time.Millisecond),
					overallRate, totalElapsed.Round(time.Millisecond))
				lastStep = now
			}
		}
	}

	log.Printf("all keys stored in %s", time.Since(start).Round(time.Millisecond))
	snapStart := time.Now()
	log.Print("snap")
	if _, err := tree.Snap(1); err != nil {
		log.Fatal(err)
	}
	log.Printf("snap finished in %s", time.Since(snapStart).Round(time.Millisecond))
	syncStart := time.Now()
	log.Print("sync")
	if err := tree.Sync(); err != nil {
		log.Fatal(err)
	}
	log.Printf("sync finished in %s", time.Since(syncStart).Round(time.Millisecond))
	log.Printf("total elapsed: %s", time.Since(start).Round(time.Millisecond))
	log.Print("done")
}
