// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mpt

import (
	"bytes"
	"encoding/binary"
	"slices"
)

// SetBatch adds or updates the given pre-sorted key-value pairs in the in-memory tree.
func (t *memTree) SetBatch(changes []KeyVal) error {
	if t.err != nil {
		return t.err
	}
	if err := checkBatch(changes); err != nil {
		return err
	}
	if len(changes) == 0 {
		return nil
	}
	t.dirty = true
	t.exact = false
	for _, kv := range changes {
		if t.root == nil {
			t.root = &memNode{key: kv.Key, val: kv.Val}
		} else if setChild(-1, &t.root, kv.Key, kv.Val) >= 0 {
			panic("bad add")
		}
	}
	return nil
}

// SetBatch adds or updates the given pre-sorted key-value pairs in the on-disk tree.
//
// Unlike calling [diskTree.Set] in a loop—which walks from the root and recomputes
// and journals O(depth) inner hashes per key—SetBatch traverses the modified frontier
// of the trie once over the sorted batch, computing and journaling each modified
// inner node's hash exactly once bottom-up.
func (t *diskTree) SetBatch(changes []KeyVal) error {
	t.mmu.RLock()
	defer t.mmu.RUnlock()

	if t.err != nil {
		return t.err
	}
	if err := checkBatch(changes); err != nil {
		return err
	}
	if len(changes) == 0 {
		return nil
	}

	if t.hdr().exact() {
		if err := t.hdr().setExact(t, false); err != nil {
			return err
		}
		if err := t.pmem.Sync(); err != nil {
			return t.broken(err)
		}
	}

	oldRoot := t.hdr().root()
	if oldRoot == 0 {
		newRoot, err := t.buildEmptyBatch(changes)
		if err != nil {
			return err
		}
		rootNode, err := t.child(newRoot)
		if err != nil {
			return err
		}
		// Set dirty and root together inside a group so a crash during a
		// multi-frame cold build never leaves dirty=true with root=0.
		if err := t.pmem.BeginGroup(); err != nil {
			return t.broken(err)
		}
		if !t.hdr().dirty() {
			if err := t.hdr().setDirty(t, true); err != nil {
				t.pmem.EndGroup()
				return err
			}
		}
		if err := t.hdr().setRoot(t, rootNode); err != nil {
			t.pmem.EndGroup()
			return err
		}
		if err := t.pmem.EndGroup(); err != nil {
			return t.broken(err)
		}
		_ = t.check // t.check()
		return nil
	}

	if !t.hdr().dirty() {
		if err := t.hdr().setDirty(t, true); err != nil {
			return err
		}
	}
	newRoot, _, err := t.setBatchSubtree(oldRoot, -1, changes, nil)
	if err != nil {
		return err
	}
	if newRoot != oldRoot {
		rootNode, err := t.child(newRoot)
		if err != nil {
			return err
		}
		if err := t.hdr().setRoot(t, rootNode); err != nil {
			return err
		}
	}
	_ = t.check // t.check()
	return nil
}

func (t *diskTree) getProbeBuf() []byte {
	n := len(t.bufPool)
	if n == 0 {
		return nil
	}
	b := t.bufPool[n-1]
	t.bufPool[n-1] = nil
	t.bufPool = t.bufPool[:n-1]
	return b
}

func (t *diskTree) putProbeBuf(b []byte) {
	if b != nil {
		t.bufPool = append(t.bufPool, b)
	}
}

// initClean initializes node n with leaf (key, val), split bit, children,
// precomputed inner hash ihash, and dirty=false in a single atomic mutation.
func (n *diskNode) initClean(t *diskTree, key Key, val Val, bit int, left, right addr, ihash Hash) error {
	leaf := t.pmem.DiskSize()
	if err := appendLeaf(t, key, val); err != nil {
		return err
	}

	var buf [nodeSize]byte
	binary.BigEndian.PutUint16(buf[nodeBitDirty:], uint16(bit)<<1) // dirty=0
	putAddr(buf[nodeLeft:], left)
	putAddr(buf[nodeRight:], right)
	putAddr(buf[nodeLeaf:], addr(leaf))
	copy(buf[nodeIHash:], ihash[:])
	return t.mutate(n[:], buf[:])
}

// buildEmptyBatch constructs a complete threaded Patricia trie for sorted changes
// (len(changes) >= 1) when the tree is initially empty.
func (t *diskTree) buildEmptyBatch(changes []KeyVal) (addr, error) {
	first := changes[0]
	nilNode, err := t.newNode()
	if err != nil {
		return 0, err
	}
	if err := nilNode.initClean(t, first.Key, first.Val, 0, 0, 0, Hash{}); err != nil {
		return 0, err
	}
	nilAddr := t.addr(nilNode)
	if len(changes) == 1 {
		return nilAddr, nil
	}

	mid := batchItem{
		key:  prefix(first.Key, maxKeyBits),
		hash: hashLeaf(first.Key, first.Val),
		addr: nilAddr,
	}
	res, err := t.mergeAround(nil, mid, changes[1:])
	if err != nil {
		return 0, err
	}
	return res.addr, nil
}

// probedLeaf caches the single leaf read along a memory-only lookup spine so
// every ancestor along that spine learns its subtree prefix without extra I/O,
// and the terminal leaf's value/buffer is already loaded for update or hashing.
type probedLeaf struct {
	nodeAddr addr
	leafOff  int64
	rawBuf   []byte
	key      Key
	val      Val
}

func (t *diskTree) probeSubtree(startAddr addr, pbit int, targetKey Key) (probedLeaf, error) {
	curAddr := startAddr
	curPbit := pbit
	for {
		cur, err := t.child(curAddr)
		if err != nil {
			return probedLeaf{}, err
		}
		nbit := cur.bit()
		if nbit <= curPbit {
			leafOff := int64(cur.leaf())
			buf, key, val, err := loadLeafInto(t, t.getProbeBuf(), leafOff, true)
			if err != nil {
				return probedLeaf{}, err
			}
			return probedLeaf{
				nodeAddr: curAddr,
				leafOff:  leafOff,
				rawBuf:   buf,
				key:      key,
				val:      val,
			}, nil
		}
		if targetKey.bit(nbit) == 0 {
			curAddr = cur.left()
		} else {
			curAddr = cur.right()
		}
		curPbit = nbit
	}
}

func (t *diskTree) setBatchSubtree(a addr, pbit int, batch []KeyVal, probed *probedLeaf) (addr, Hash, error) {
	n, err := t.child(a)
	if err != nil {
		return 0, Hash{}, err
	}
	nbit := n.bit()

	if probed == nil {
		pl, err := t.probeSubtree(a, pbit, batch[0].Key)
		if err != nil {
			return 0, Hash{}, err
		}
		defer t.putProbeBuf(pl.rawBuf)
		probed = &pl
	}

	bits := nbit
	if nbit <= pbit {
		bits = maxKeyBits
	}
	pkey := prefix(probed.key, bits)
	before, inside, after := partitionByPrefix(batch, pkey)

	var mid batchItem
	if bits == maxKeyBits {
		// Leaf node: probed must be this exact leaf.
		if probed.nodeAddr != a {
			return 0, Hash{}, t.broken(errCorrupt)
		}
		var h Hash
		if len(inside) == 1 {
			newVal := inside[0].Val
			if err := n.setValLoaded(t, probed.leafOff, probed.rawBuf, probed.key, newVal); err != nil {
				return 0, Hash{}, err
			}
			h = hashLeaf(probed.key, newVal)
		} else {
			h = hashLeaf(probed.key, probed.val)
		}
		mid = batchItem{
			key:  pkey,
			hash: h,
			addr: a,
		}
	} else if len(inside) == 0 {
		// Inner node with no modifications inside its subtree; only before/after splices exist.
		var h Hash
		if n.dirty() {
			h, err = n.rehash(t, pbit, false)
			if err != nil {
				return 0, Hash{}, err
			}
		} else {
			h = n.ihash()
		}
		mid = batchItem{
			key:  pkey,
			hash: h,
			addr: a,
		}
	} else {
		// Inner node with modifications inside its subtree.
		// Mark n dirty before mutating descendants so any intermediate
		// pmem state flushed prior to unwind remains hash-consistent.
		if !n.dirty() {
			if err := n.setDirty(t, true); err != nil {
				return 0, Hash{}, err
			}
		}
		inLeft, inRight := splitAtBit(inside, nbit)
		probedRight := probed.key.bit(nbit) != 0

		origLeft := n.left()
		origRight := n.right()

		var lAddr addr
		var lHash Hash
		if len(inLeft) > 0 {
			var lp *probedLeaf
			if !probedRight {
				lp = probed
			}
			lAddr, lHash, err = t.setBatchSubtree(origLeft, nbit, inLeft, lp)
			if err != nil {
				return 0, Hash{}, err
			}
		} else {
			lAddr = origLeft
			lnode, err := t.child(lAddr)
			if err != nil {
				return 0, Hash{}, err
			}
			if lnode.dirty() {
				lHash, err = lnode.rehash(t, nbit, false)
			} else {
				lHash, err = lnode.hash(t, nbit)
			}
			if err != nil {
				return 0, Hash{}, err
			}
		}

		var rAddr addr
		var rHash Hash
		if len(inRight) > 0 {
			var rp *probedLeaf
			if probedRight {
				rp = probed
			}
			rAddr, rHash, err = t.setBatchSubtree(origRight, nbit, inRight, rp)
			if err != nil {
				return 0, Hash{}, err
			}
		} else {
			rAddr = origRight
			rnode, err := t.child(rAddr)
			if err != nil {
				return 0, Hash{}, err
			}
			if rnode.dirty() {
				rHash, err = rnode.rehash(t, nbit, false)
			} else {
				rHash, err = rnode.hash(t, nbit)
			}
			if err != nil {
				return 0, Hash{}, err
			}
		}

		// Re-fetch n after recursive calls in case t.mem expanded or n's own
		// threaded leaf slot was reallocated by a descendant value growth.
		n, err = t.child(a)
		if err != nil {
			return 0, Hash{}, err
		}
		h := hashInner(nbit, lHash, rHash)
		var buf [nodeSize]byte
		binary.BigEndian.PutUint16(buf[nodeBitDirty:], uint16(nbit)<<1) // dirty=0
		putAddr(buf[nodeLeft:], lAddr)
		putAddr(buf[nodeRight:], rAddr)
		copy(buf[nodeLeaf:nodeLeaf+addrSize], n[nodeLeaf:nodeLeaf+addrSize])
		copy(buf[nodeIHash:], h[:])
		if err := t.mutate(n[:], buf[:]); err != nil {
			return 0, Hash{}, err
		}
		mid = batchItem{
			key:  pkey,
			hash: h,
			addr: a,
		}
	}

	if len(before) == 0 && len(after) == 0 {
		return mid.addr, mid.hash, nil
	}
	res, err := t.mergeAround(before, mid, after)
	if err != nil {
		return 0, Hash{}, err
	}
	return res.addr, res.hash, nil
}

// A batchItem represents a subtree on the reduction stack during mergeAround.
// Either the item is Complete (addr != 0 && !hasHole), or it has exactly one
// unassigned new leaf awaiting an ancestor inner node owner (addr == 0 for a
// single leaf kv, or addr != 0 && hasHole for a merged new subtrie).
type batchItem struct {
	key         keyPrefix
	hash        Hash
	addr        addr
	kv          KeyVal
	hasHole     bool
	holePtrAddr addr
	holeKV      KeyVal
}

func (it batchItem) needsOwner() bool {
	return it.addr == 0 || it.hasHole
}

func (it batchItem) unassignedKV() KeyVal {
	if it.addr == 0 {
		return it.kv
	}
	return it.holeKV
}

// mergeAround folds sorted new keys before, the complete existing/updated
// subtree mid, and sorted new keys after into a single complete subtree.
func (t *diskTree) mergeAround(before []KeyVal, mid batchItem, after []KeyVal) (batchItem, error) {
	var stackBuf [64]batchItem
	s := stackBuf[:0]

	push := func(it batchItem) error {
		s = append(s, it)
		for len(s) >= 3 && s[len(s)-3].key.overlap(s[len(s)-2].key) > s[len(s)-2].key.overlap(s[len(s)-1].key) {
			m, err := t.mergeBatchItems(s[len(s)-3], s[len(s)-2])
			if err != nil {
				return err
			}
			s[len(s)-3] = m
			s[len(s)-2] = s[len(s)-1]
			s = s[:len(s)-1]
		}
		return nil
	}

	for _, kv := range before {
		if err := push(batchItem{
			key:  prefix(kv.Key, maxKeyBits),
			hash: hashLeaf(kv.Key, kv.Val),
			kv:   kv,
		}); err != nil {
			return batchItem{}, err
		}
	}
	if err := push(mid); err != nil {
		return batchItem{}, err
	}
	for _, kv := range after {
		if err := push(batchItem{
			key:  prefix(kv.Key, maxKeyBits),
			hash: hashLeaf(kv.Key, kv.Val),
			kv:   kv,
		}); err != nil {
			return batchItem{}, err
		}
	}

	for len(s) >= 2 {
		m, err := t.mergeBatchItems(s[len(s)-2], s[len(s)-1])
		if err != nil {
			return batchItem{}, err
		}
		s[len(s)-2] = m
		s = s[:len(s)-1]
	}
	if s[0].needsOwner() {
		panic("mpt: internal error: unresolved hole after mergeAround")
	}
	return s[0], nil
}

func (t *diskTree) mergeBatchItems(x, y batchItem) (batchItem, error) {
	b := x.key.overlap(y.key)
	h := hashInner(b, x.hash, y.hash)

	u, err := t.newNode()
	if err != nil {
		return batchItem{}, err
	}
	uAddr := t.addr(u)

	xNeeds := x.needsOwner()
	yNeeds := y.needsOwner()

	switch {
	case !xNeeds && yNeeds:
		// u owns y's unassigned key; result is Complete.
		ykv := y.unassignedKV()
		rightAddr := uAddr
		if y.addr != 0 {
			rightAddr = y.addr
			if err := t.setAddrAt(y.holePtrAddr, uAddr); err != nil {
				return batchItem{}, err
			}
		}
		if err := u.initClean(t, ykv.Key, ykv.Val, b, x.addr, rightAddr, h); err != nil {
			return batchItem{}, err
		}
		return batchItem{
			key:  x.key.truncate(b),
			hash: h,
			addr: uAddr,
		}, nil

	case xNeeds && !yNeeds:
		// u owns x's unassigned key; result is Complete.
		xkv := x.unassignedKV()
		leftAddr := uAddr
		if x.addr != 0 {
			leftAddr = x.addr
			if err := t.setAddrAt(x.holePtrAddr, uAddr); err != nil {
				return batchItem{}, err
			}
		}
		if err := u.initClean(t, xkv.Key, xkv.Val, b, leftAddr, y.addr, h); err != nil {
			return batchItem{}, err
		}
		return batchItem{
			key:  x.key.truncate(b),
			hash: h,
			addr: uAddr,
		}, nil

	case xNeeds && yNeeds:
		// u owns y's unassigned key; x's unassigned key propagates up as a hole.
		ykv := y.unassignedKV()
		rightAddr := uAddr
		if y.addr != 0 {
			rightAddr = y.addr
			if err := t.setAddrAt(y.holePtrAddr, uAddr); err != nil {
				return batchItem{}, err
			}
		}
		leftAddr := x.addr // 0 if x is a single unassigned leaf
		holePtr := x.holePtrAddr
		holeKV := x.holeKV
		if x.addr == 0 {
			holePtr = uAddr + nodeLeft
			holeKV = x.kv
		}
		if err := u.initClean(t, ykv.Key, ykv.Val, b, leftAddr, rightAddr, h); err != nil {
			return batchItem{}, err
		}
		return batchItem{
			key:         x.key.truncate(b),
			hash:        h,
			addr:        uAddr,
			hasHole:     true,
			holePtrAddr: holePtr,
			holeKV:      holeKV,
		}, nil

	default:
		panic("mpt: internal error: two complete items in mergeAround")
	}
}

// partitionByPrefix splits strictly ascending batch into keys < p, == p, and > p.
func partitionByPrefix(batch []KeyVal, p keyPrefix) (before, inside, after []KeyVal) {
	n := len(batch)
	if n == 0 {
		return nil, nil, nil
	}
	cFirst := prefix(batch[0].Key, p.bits).compare(p)
	if cFirst > 0 {
		return nil, nil, batch
	}
	cLast := prefix(batch[n-1].Key, p.bits).compare(p)
	if cLast < 0 {
		return batch, nil, nil
	}
	if cFirst == 0 && cLast == 0 {
		return nil, batch, nil
	}

	inStart := 0
	if cFirst < 0 {
		inStart, _ = slices.BinarySearchFunc(batch, p, func(kv KeyVal, target keyPrefix) int {
			return prefix(kv.Key, target.bits).compare(target)
		})
	}
	if p.bits == maxKeyBits {
		if inStart < n && bytes.Equal(batch[inStart].Key, p.full) {
			return batch[:inStart], batch[inStart : inStart+1], batch[inStart+1:]
		}
		return batch[:inStart], nil, batch[inStart:]
	}
	inEnd := n
	if cLast > 0 {
		off, _ := slices.BinarySearchFunc(batch[inStart:], p, func(kv KeyVal, target keyPrefix) int {
			if prefix(kv.Key, target.bits).compare(target) <= 0 {
				return -1
			}
			return 1
		})
		inEnd = inStart + off
	}
	return batch[:inStart], batch[inStart:inEnd], batch[inEnd:]
}

// splitAtBit splits inside (all sharing bits 0..b-1, sorted by Key.Compare)
// into keys with bit(b) == 0 and keys with bit(b) == 1.
func splitAtBit(inside []KeyVal, b int) (left, right []KeyVal) {
	n := len(inside)
	if n == 0 || inside[0].Key.bit(b) != 0 {
		return nil, inside
	}
	if inside[n-1].Key.bit(b) == 0 {
		return inside, nil
	}
	i, _ := slices.BinarySearchFunc(inside, b, func(kv KeyVal, bitIdx int) int {
		return kv.Key.bit(bitIdx) - 1
	})
	return inside[:i], inside[i:]
}
