// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/etensor/tensor"
)

// ProbeEnv generates sentences using a grammar that is parsed from a
// text file.  The core of the grammar is rules with various items
// chosen at random during generation -- these items can be
// more rules terminal tokens.
type ProbeEnv struct {
	// name of this environment
	Name string

	// list of words used for activating state units according to index
	Words []string

	// current sentence activation state
	WordState tensor.Float32

	// trial is the step counter within sequence - how many steps taken
	// within current sequence -- it resets to 0 at start of each sequence.
	Trial env.Counter `view:"inline"`
}

func (ev *ProbeEnv) Label() string { return ev.Name }

func (ev *ProbeEnv) State(element string) tensor.Tensor {
	switch element {
	case "Input":
		return &ev.WordState
	}
	return nil
}

func (ev *ProbeEnv) Init(run int) {
	ev.Trial.Scale = etime.Trial
	ev.Trial.Max = len(ev.Words)
	ev.Trial.Init()
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.WordState.SetShape([]int{len(ev.Words)})
}

// String returns the current state as a string
func (ev *ProbeEnv) String() string {
	if ev.Trial.Cur < len(ev.Words) {
		return fmt.Sprintf("%s", ev.Words[ev.Trial.Cur])
	} else {
		return ""
	}
}

// RenderState renders the current state
func (ev *ProbeEnv) RenderState() {
	ev.WordState.SetZeros()
	if ev.Trial.Cur > 0 && ev.Trial.Cur < len(ev.Words) {
		ev.WordState.SetFloat1D(ev.Trial.Cur, 1)
	}
}

func (ev *ProbeEnv) Step() bool {
	ev.Trial.Incr()
	ev.RenderState()
	return true
}

func (ev *ProbeEnv) Action(element string, input tensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*ProbeEnv)(nil)
