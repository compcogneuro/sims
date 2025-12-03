// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"log"
	"math/rand"
	"strings"

	"cogentcore.org/lab/base/randx"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/esg"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/etensor/tensor"
)

// SentGenEnv generates sentences using a grammar that is parsed from a
// text file.  The core of the grammar is rules with various items
// chosen at random during generation -- these items can be
// more rules terminal tokens.
type SentGenEnv struct {

	// name of this environment
	Name string

	// core sent-gen rules -- loaded from a grammar / rules file -- Gen() here generates one sentence
	Rules esg.Rules

	// probability of generating passive sentence forms
	PPassive float64

	// translate unambiguous words into ambiguous words
	WordTrans map[string]string

	// list of words used for activating state units according to index
	Words []string

	// map of words onto index in Words list
	WordMap map[string]int

	// list of roles used for activating state units according to index
	Roles []string

	// map of roles onto index in Roles list
	RoleMap map[string]int

	// list of filler concepts used for activating state units according to index
	Fillers []string

	// map of roles onto index in Words list
	FillerMap map[string]int

	// ambiguous verbs
	AmbigVerbs []string

	// ambiguous nouns
	AmbigNouns []string

	// map of ambiguous verbs
	AmbigVerbsMap map[string]int

	// map of ambiguous nouns
	AmbigNounsMap map[string]int

	// original current sentence as generated from Rules
	CurSentOrig []string

	// current sentence, potentially transformed to passive form
	CurSent []string

	// number of ambiguous nouns
	NAmbigNouns int

	// number of ambiguous verbs (0 or 1)
	NAmbigVerbs int

	// generated sequence of sentence inputs including role-filler queries
	SentInputs [][]string

	// current index within sentence inputs
	SentIndex env.CurPrvInt

	// current question type -- from 4th value of SentInputs
	QType string

	// current sentence activation state
	WordState tensor.Float32

	// current role query activation state
	RoleState tensor.Float32

	// current filler query activation state
	FillerState tensor.Float32

	// sequence counter within epoch
	Seq env.Counter `view:"inline"`

	// tick counter within sequence
	Tick env.Counter `view:"inline"`

	// trial is the step counter within sequence - how many steps taken within current sequence -- it resets to 0 at start of each sequence
	Trial env.Counter `view:"inline"`
}

func (ev *SentGenEnv) Label() string { return ev.Name }

// InitTMat initializes matrix and labels to given size
func (ev *SentGenEnv) Validate() error {
	ev.Rules.Validate()
	return nil
}

func (ev *SentGenEnv) State(element string) tensor.Tensor {
	switch element {
	case "Input":
		return &ev.WordState
	case "Role":
		return &ev.RoleState
	case "Filler":
		return &ev.FillerState
	}
	return nil
}

func (ev *SentGenEnv) OpenRulesFromAsset(fnm string) {
	ab, err := content.ReadFile(fnm)
	if err != nil {
		log.Println(err)
	}
	ev.Rules.ReadRules(bytes.NewBuffer(ab))
}

func (ev *SentGenEnv) Init(run int) {
	ev.Seq.Scale = etime.Sequence
	ev.Tick.Scale = etime.Tick
	ev.Trial.Scale = etime.Trial
	ev.Seq.Init()
	ev.Tick.Init()
	ev.Trial.Init()
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.SentIndex.Set(-1)

	ev.Rules.Init()
	ev.MapsFmWords()

	ev.WordState.SetShape([]int{len(ev.Words)})
	ev.RoleState.SetShape([]int{len(ev.Roles)})
	ev.FillerState.SetShape([]int{len(ev.Fillers)})
}

func (ev *SentGenEnv) MapsFmWords() {
	ev.WordMap = make(map[string]int, len(ev.Words))
	for i, wrd := range ev.Words {
		ev.WordMap[wrd] = i
	}
	ev.RoleMap = make(map[string]int, len(ev.Roles))
	for i, wrd := range ev.Roles {
		ev.RoleMap[wrd] = i
	}
	ev.FillerMap = make(map[string]int, len(ev.Fillers))
	for i, wrd := range ev.Fillers {
		ev.FillerMap[wrd] = i
	}
	ev.AmbigVerbsMap = make(map[string]int, len(ev.AmbigVerbs))
	for i, wrd := range ev.AmbigVerbs {
		ev.AmbigVerbsMap[wrd] = i
	}
	ev.AmbigNounsMap = make(map[string]int, len(ev.AmbigNouns))
	for i, wrd := range ev.AmbigNouns {
		ev.AmbigNounsMap[wrd] = i
	}
}

// CurInputs returns current inputs triple from SentInputs
func (ev *SentGenEnv) CurInputs() []string {
	if ev.SentIndex.Cur >= 0 && ev.SentIndex.Cur < len(ev.SentInputs) {
		return ev.SentInputs[ev.SentIndex.Cur]
	}
	return nil
}

// String returns the current state as a string
func (ev *SentGenEnv) String() string {
	cur := ev.CurInputs()
	if cur != nil {
		return fmt.Sprintf("%s %s=%s %s", cur[0], cur[1], cur[2], cur[3])
	}
	return ""
}

// NextSent generates the next sentence and all the queries for it
func (ev *SentGenEnv) NextSent() {
	// ev.Rules.Trace = true
	ev.CurSent = ev.Rules.Gen()
	// fmt.Printf("%v\n", ev.CurSent)
	ev.Rules.States.TrimQualifiers()
	ev.SentStats()
	ev.SentIndex.Set(0)
	if cs, has := ev.Rules.States["Case"]; has {
		if cs == "Passive" {
			ev.SentSeqPassive()
		} else {
			ev.SentSeqActive()
		}
	} else {
		if randx.BoolP(ev.PPassive) {
			ev.SentSeqPassive()
		} else {
			ev.SentSeqActive()
		}
	}
}

// TransWord gets the translated word
func (ev *SentGenEnv) TransWord(word string) string {
	word = strings.ToLower(word)
	if tr, has := ev.WordTrans[word]; has {
		return tr
	}
	return word
}

// SentStats computes stats on sentence (ambig words)
func (ev *SentGenEnv) SentStats() {
	ev.NAmbigNouns = 0
	ev.NAmbigVerbs = 0
	for _, wrd := range ev.CurSent {
		wrd = ev.TransWord(wrd)
		if _, has := ev.AmbigVerbsMap[wrd]; has {
			ev.NAmbigVerbs++
		}
		if _, has := ev.AmbigNounsMap[wrd]; has {
			ev.NAmbigNouns++
		}
	}
}

// CheckWords reports errors if words not found, if not empty
func (ev *SentGenEnv) CheckWords(wrd, role, fill string) []error {
	var errs []error
	if _, ok := ev.WordMap[wrd]; !ok {
		errs = append(errs, fmt.Errorf("word not found in WordMap: %s, sent: %v", wrd, ev.CurSent))
	}
	if _, ok := ev.RoleMap[role]; !ok {
		errs = append(errs, fmt.Errorf("word not found in RoleMap: %s, sent: %v", role, ev.CurSent))
	}
	if _, ok := ev.FillerMap[fill]; !ok {
		errs = append(errs, fmt.Errorf("word not found in FillerMap: %s, sent: %v", fill, ev.CurSent))
	}
	if errs != nil {
		for _, err := range errs {
			fmt.Println(err)
		}
	}
	return errs
}

func (ev *SentGenEnv) NewInputs() {
	ev.SentInputs = make([][]string, 0, 16)
}

// AddRawInput adds raw input
func (ev *SentGenEnv) AddRawInput(word, role, fill, stat string) {
	ev.SentInputs = append(ev.SentInputs, []string{word, role, fill, stat})
}

// AddInput adds a new input with given sentence index word and role query
// stat is an extra status var: "revq" or "curq" (review question, vs. current question)
func (ev *SentGenEnv) AddInput(sidx int, role string, stat string) {
	wrd := ev.TransWord(ev.CurSent[sidx])
	fill := ev.Rules.States[role]
	ev.CheckWords(wrd, role, fill)
	ev.AddRawInput(wrd, role, fill, stat)
}

// AddQuestion adds a new input with 'question' word and role query
// automatically marked as a "revq"
func (ev *SentGenEnv) AddQuestion(role string) {
	wrd := "question"
	fill := ev.Rules.States[role]
	ev.CheckWords(wrd, role, fill)
	ev.AddRawInput(wrd, role, fill, "revq")
}

// SentSeqActive active form sentence sequence, with incremental review questions
func (ev *SentGenEnv) SentSeqActive() {
	ev.NewInputs()
	ev.AddRawInput("start", "Action", "None", "curq") // start question helps in long run!
	mod := ev.Rules.States["Mod"]
	seq := []string{"Agent", "Action", "Patient", mod}
	for si := 0; si < 3; si++ {
		sq := seq[si]
		ev.AddInput(si, sq, "curq")
		switch si { // these additional questions are key for revq perf
		case 1:
			ev.AddInput(si, "Agent", "revq")
		case 2:
			ev.AddInput(si, "Action", "revq")
		}
	}
	slen := len(ev.CurSent)
	if slen == 3 {
		return
	}
	// get any modifier words with random query
	for si := 3; si < slen-1; si++ {
		ri := rand.Intn(3) // choose a role to query at random
		ev.AddInput(si, seq[ri], "revq")
	}
	ev.AddInput(slen-1, mod, "curq")
	ri := rand.Intn(3) // choose a role to query at random
	if fq, has := ev.Rules.States["FinalQ"]; has {
		for i := range seq {
			if seq[i] == fq {
				ri = i
				break
			}
		}
	}
	ev.AddInput(slen-1, seq[ri], "revq")
}

// SentSeqPassive passive form sentence sequence, with incremental review questions
func (ev *SentGenEnv) SentSeqPassive() {
	ev.NewInputs()
	ev.AddRawInput("start", "Action", "None", "curq") // start question helps in long run!
	mod := ev.Rules.States["Mod"]
	seq := []string{"Agent", "Action", "Patient", mod}
	ev.AddInput(2, "Patient", "curq") // 2 = patient word in active form
	ev.AddRawInput("was", "Patient", ev.Rules.States["Patient"], "revq")
	ev.AddInput(1, "Action", "curq") // 1 = action word in active form
	ev.AddRawInput("by", "Action", ev.Rules.States["Action"], "revq")
	ev.AddInput(0, "Agent", "curq") // 0 = agent word in active form
	// note: we already get review questions for free with was and by
	// get any modifier words with random query
	slen := len(ev.CurSent)
	for si := 3; si < slen-1; si++ {
		ri := rand.Intn(3) // choose a role to query at random
		ev.AddInput(si, seq[ri], "revq")
	}
	ev.AddInput(slen-1, mod, "curq")
	ri := rand.Intn(3) // choose a role to query at random
	// ev.AddQuestion(seq[ri])
	ev.AddInput(slen-1, seq[ri], "revq")
}

// RenderState renders the current state
func (ev *SentGenEnv) RenderState() {
	ev.WordState.SetZeros()
	ev.RoleState.SetZeros()
	ev.FillerState.SetZeros()
	cur := ev.CurInputs()
	if cur == nil {
		return
	}
	widx := ev.WordMap[cur[0]]
	ev.WordState.SetFloat1D(widx, 1)
	ridx := ev.RoleMap[cur[1]]
	ev.RoleState.SetFloat1D(ridx, 1)
	fidx := ev.FillerMap[cur[2]]
	ev.FillerState.SetFloat1D(fidx, 1)
	ev.QType = cur[3]
}

// NextState generates the next inputs
func (ev *SentGenEnv) NextState() {
	if ev.SentIndex.Cur < 0 {
		ev.NextSent()
	} else {
		ev.SentIndex.Incr()
	}
	if ev.SentIndex.Cur >= len(ev.SentInputs) {
		ev.NextSent()
	}
	ev.RenderState()
}

func (ev *SentGenEnv) Step() bool {
	ev.NextState()
	ev.Trial.Incr()
	ev.Tick.Incr()
	if ev.SentIndex.Cur == 0 {
		ev.Tick.Init()
		ev.Seq.Incr()
	}
	return true
}

func (ev *SentGenEnv) Action(element string, input tensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*SentGenEnv)(nil)
