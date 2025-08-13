// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// catsdogs: This project explores a simple **semantic network** intended
// to represent a (very small) set of relationships among different
// features used to represent a set of entities in the world.
// In our case, we represent some features of cats and dogs:
// their color, size, favorite food, and favorite toy.
package catsdogs

//go:generate core generate -add-types -add-funcs

import (
	"embed"
	"fmt"
	"io/fs"
	"reflect"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed cats_dogs_pats.tsv cats_dogs.wts
var embedfs embed.FS

//go:embed README.md
var readme embed.FS

// Modes are the looping modes (Stacks) for running and statistics.
type Modes int32 //enums:enum
const (
	Train Modes = iota
	Test
)

// Levels are the looping levels for running and statistics.
type Levels int32 //enums:enum
const (
	Cycle Levels = iota
	Trial
	Epoch
	Run
	Expt
)

// StatsPhase is the phase of stats processing for given mode, level.
// Accumulated values are reset at Start, added each Step.
type StatsPhase int32 //enums:enum
const (
	Start StatsPhase = iota
	Step
)

// see params.go for params, config.go for config

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// the patterns to use
	Patterns *table.Table `new-window:"+" display:"no-inline"`

	// simulation configuration parameters -- set by .toml config file and / or args
	Config *Config `new-window:"+"`

	// Net is the network: click to view / edit parameters for layers, paths, etc.
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// Params manages network parameter setting.
	Params leabra.Params `display:"inline"`

	// Loops are the control loops for running the sim, in different Modes
	// across stacks of Levels.
	Loops *looper.Stacks `display:"-"`

	// Envs provides mode-string based storage of environments.
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// TestUpdate has Test mode netview update parameters.
	TestUpdate leabra.NetViewUpdate `display:"inline"`

	// Root is the root tensorfs directory, where all stats and other misc sim data goes.
	Root *tensorfs.Node `display:"-"`

	// Stats has the stats directory within Root.
	Stats *tensorfs.Node `display:"-"`

	// Current has the current stats values within Stats.
	Current *tensorfs.Node `display:"-"`

	// StatFuncs are statistics functions called at given mode and level,
	// to perform all stats computations. phase = Start does init at start of given level,
	// and all intialization / configuration (called during Init too).
	StatFuncs []func(mode Modes, level Levels, phase StatsPhase) `display:"-"`

	// GUI manages all the GUI elements
	GUI egui.GUI `display:"-"`

	// RandSeeds is a list of random seeds to use for each run.
	RandSeeds randx.Seeds `display:"-"`
}

func (ss *Sim) SetConfig(cfg *Config) { ss.Config = cfg }
func (ss *Sim) Body() *core.Body      { return ss.GUI.Body }

func (ss *Sim) ConfigSim() {
	ss.Root, _ = tensorfs.NewDir("Root")
	tensorfs.CurRoot = ss.Root
	ss.Net = leabra.NewNetwork(ss.Config.Name)
	ss.Params.Config(LayerParams, PathParams, ss.Config.Params.Sheet, ss.Config.Params.Tag, reflect.ValueOf(ss))
	ss.Patterns = &table.Table{}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.OpenPatterns()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLoops()
	ss.ConfigStats()
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var tst *env.FixedTable
	if len(ss.Envs) == 0 {
		tst = &env.FixedTable{}
	} else {
		tst = ss.Envs.ByMode(Test).(*env.FixedTable)
	}

	tst.Name = Test.String()
	tst.Config(table.NewView(ss.Patterns))
	tst.Sequential = true
	tst.Validate()

	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	name := net.AddLayer2D("Name", leabra.InputLayer, 1, 10)
	iden := net.AddLayer2D("Identity", leabra.InputLayer, 1, 10)
	color := net.AddLayer2D("Color", leabra.InputLayer, 1, 4)
	food := net.AddLayer2D("FavoriteFood", leabra.InputLayer, 1, 4)
	size := net.AddLayer2D("Size", leabra.InputLayer, 1, 3)
	spec := net.AddLayer2D("Species", leabra.InputLayer, 1, 2)
	toy := net.AddLayer2D("FavoriteToy", leabra.InputLayer, 1, 4)

	name.AddClass("Id") // share params
	iden.AddClass("Id")

	one2one := paths.NewOneToOne()
	full := paths.NewFull()

	net.BidirConnectLayers(name, iden, one2one)
	net.BidirConnectLayers(color, iden, full)
	net.BidirConnectLayers(food, iden, full)
	net.BidirConnectLayers(size, iden, full)
	net.BidirConnectLayers(spec, iden, full)
	net.BidirConnectLayers(toy, iden, full)

	iden.PlaceAbove(name)
	color.PlaceAbove(iden)
	// gend.Pos.XAlign = relpos.Right
	food.PlaceAbove(iden)
	food.Pos.XAlign = relpos.Right
	size.PlaceAbove(color)
	spec.PlaceAbove(color)
	spec.Pos.XAlign = relpos.Right
	spec.Pos.XOffset = 2
	toy.PlaceAbove(food)
	toy.Pos.XAlign = relpos.Right
	toy.Pos.XOffset = 1

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	ss.InitWeights(net)
}

// InitWeights initializes weights.
func (ss *Sim) InitWeights(net *leabra.Network) {
	net.InitWeights()
	net.OpenWeightsFS(embedfs, "cats_dogs.wts")
}

func (ss *Sim) ApplyParams() {
	ss.Params.ApplyAll(ss.Net)
}

////////  Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	// ss.SetRunName() // TODO remove?
	ss.InitRandSeed(0)
	ss.ApplyParams()
	ss.StatsInit()
	ss.NewRun()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// NetViewUpdater returns the NetViewUpdate for given mode.
func (ss *Sim) NetViewUpdater(mode enums.Enum) *leabra.NetViewUpdate {
	if mode.Int64() == Test.Int64() {
		return &ss.TestUpdate
	}
	return &ss.TestUpdate
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	cycles := 100
	plusPhase := 50

	ev := ss.Envs.ByMode(Test).(*env.FixedTable)
	ntrls := ev.Table.NumRows()

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevel(Trial, ntrls).
		AddLevel(Cycle, cycles)

	leabra.LooperStandard(ls, ss.Net, ss.NetViewUpdater, cycles-plusPhase, cycles-1, Cycle, Trial, Train)

	ls.Stacks[Test].OnInit.Add("Init", ss.Init)

	ls.AddOnStartToLoop(Trial, "ApplyInputs", func(mode enums.Enum) {
		ss.ApplyInputs(mode.(Modes))
	})

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	if ss.Config.GUI {
		leabra.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

		ls.Stacks[Test].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
	}

	if ss.Config.Debug {
		mpi.Println(ls.DocString())
	}
	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment for given mode.
// Any other start-of-trial logic can also be put here.
func (ss *Sim) ApplyInputs(mode Modes) {
	net := ss.Net
	curModeDir := ss.Current.Dir(mode.String())
	ev := ss.Envs.ByMode(mode)

	lays := net.LayersByType(leabra.InputLayer, leabra.CompareLayer)
	net.InitExt()
	ev.Step()
	curModeDir.StringValue("TrialName", 1).SetString1D(ev.String(), 0)
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		st := ev.State(ly.Name)
		if st != nil {
			ly.ApplyExt(st)
		}
	}
	net.ApplyExts()
}

// NewRun intializes a new Run level of the model.
func (ss *Sim) NewRun() {
	ctx := ss.Net.Context()
	ss.Envs.ByMode(Test).Init(0)
	ctx.Reset()
	ss.InitWeights(ss.Net)
}

/////////  Patterns

func (ss *Sim) OpenPatterns() {
	dt := table.New()
	metadata.SetName(dt, "CatsAndDogs")
	metadata.SetDoc(dt, "Face testing patterns")
	err := dt.OpenFS(embedfs, "cats_dogs_pats.tsv", tensor.Tab)
	if errors.Log(err) != nil {
		fmt.Println(err)
	}
	ss.Patterns = dt
}

////////  Inputs

// OpenTable opens a [table.Table] from embedded content, storing
// the data in the given tensorfs directory.
func (ss *Sim) OpenTable(dir *tensorfs.Node, fsys fs.FS, fnm, name, docs string) (*table.Table, error) {
	dt := table.New()
	metadata.SetName(dt, name)
	metadata.SetDoc(dt, docs)
	err := dt.OpenFS(embedfs, fnm, tensor.Tab)
	if errors.Log(err) != nil {
		return dt, err
	}
	tensorfs.DirFromTable(dir.Dir(name), dt)
	return dt, err
}

// func (ss *Sim) OpenInputs() {
// 	dir := ss.Root.Dir("Inputs")
// 	ss.OpenTable(dir, embedfs, "easy.tsv", "Easy", "Easy patterns")
// 	ss.OpenTable(dir, embedfs, "hard.tsv", "Hard", "Hard patterns")
// 	ss.OpenTable(dir, embedfs, "impossible.tsv", "Impossible", "Impossible patterns")
// } // TODO remove

//////// Stats

// AddStat adds a stat compute function.
func (ss *Sim) AddStat(f func(mode Modes, level Levels, phase StatsPhase)) {
	ss.StatFuncs = append(ss.StatFuncs, f)
}

// StatsStart is called by Looper at the start of given level, for each iteration.
// It needs to call RunStats Start at the next level down.
// e.g., each Epoch is the start of the full set of Trial Steps.
func (ss *Sim) StatsStart(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level <= Cycle {
		return
	}
	ss.RunStats(mode, level-1, Start)
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	// if level == Cycle {
	// 	return
	// }
	ss.RunStats(mode, level, Step)
	tensorfs.DirTable(leabra.StatsNode(ss.Stats, mode, level), nil).WriteToLog()
}

// RunStats runs the StatFuncs for given mode, level and phase.
func (ss *Sim) RunStats(mode Modes, level Levels, phase StatsPhase) {
	for _, sf := range ss.StatFuncs {
		sf(mode, level, phase)
	}
	if phase == Step && ss.GUI.Tabs != nil {
		nm := mode.String() + " " + level.String() + " Plot"
		ss.GUI.Tabs.AsLab().GoUpdatePlot(nm)
		if level == Run {
			ss.GUI.Tabs.AsLab().GoUpdatePlot("Train RunAll Plot") // TODO what is train runall
		}
	}
}

// SetRunName sets the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) SetRunName() string {
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Current.StringValue("RunName", 1).SetString1D(runName, 0)
	return runName
}

// RunName returns the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) RunName() string {
	return ss.Current.StringValue("RunName", 1).String1D(0)
}

// StatsInit initializes all the stats by calling Start across all modes and levels.
func (ss *Sim) StatsInit() {
	for md, st := range ss.Loops.Stacks {
		mode := md.(Modes)
		for _, lev := range st.Order {
			level := lev.(Levels)
			ss.RunStats(mode, level, Start)
		}
	}
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(leabra.StatsNode(ss.Stats, Test, Cycle))
		tbs.PlotTensorFS(leabra.StatsNode(ss.Stats, Test, Trial))
		tbs.SelectTabIndex(idx)
	}
}

// ConfigStats handles configures functions to do all stats computation
// in the tensorfs system.
func (ss *Sim) ConfigStats() {
	net := ss.Net
	ss.Stats = ss.Root.Dir("Stats")
	ss.Current = ss.Stats.Dir("Current")

	ss.SetRunName()

	// last arg(s) are levels to exclude
	counterFunc := leabra.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		counterFunc(mode, level, phase == Start)
	})
	// runNameFunc := leabra.StatRunName(ss.Stats, ss.Current, ss.Loops, net, Trial)
	// ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
	// 	runNameFunc(mode, level, phase == Start)
	// })
	trialNameFunc := leabra.StatTrialName(ss.Stats, ss.Current, ss.Loops, net, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trialNameFunc(mode, level, phase == Start)
	})

	// up to a point, it is good to use loops over stats in one function,
	// to reduce repetition of boilerplate.
	statNames := []string{"Harmony"}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		for _, name := range statNames {
			di := 0
			ndata := 1
			modeDir := ss.Stats.Dir(mode.String())
			curModeDir := ss.Current.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			subDir := modeDir.Dir((level - 1).String()) // note: will fail for Cycle
			tsr := levelDir.Float64(name)
			var stat float64
			if phase == Start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					switch name {
					case "Harmony":
						if level == Cycle {
							s.On = true
						}
					}
				})
				continue
			}
			switch level {
			case Cycle:
				var stat float64
				switch name {
				case "Harmony":
					stat = float64(ss.Harmony(ss.Net))
				default:
					stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				}
				curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
				tsr.AppendRowFloat(stat)
			default: // Expt
				stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			}
		}
	})

	stateFunc := leabra.StatLayerState(ss.Stats, net, Test, Trial, true, "Act", "Name", "Identity", "Color", "FavoriteFood", "Size", "Species", "FavoriteToy")
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		stateFunc(mode, level, phase == Start)
	})
}

// Harmony computes the harmony (excitatory net input Ge * Act)
func (ss *Sim) Harmony(nt *leabra.Network) float32 {
	harm := float32(0)
	nu := 0
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		for i := range ly.Neurons {
			nrn := &(ly.Neurons[i])
			harm += nrn.Ge * nrn.Act
			nu++
		}
	}
	if nu > 0 {
		harm /= float32(nu)
	}
	return harm
}

// StatCounters returns counters string to show at bottom of netview.
func (ss *Sim) StatCounters(mode, level enums.Enum) string {
	counters := ss.Loops.Stacks[mode].CountersString()
	vu := ss.NetViewUpdater(mode)
	if vu == nil || vu.View == nil {
		return counters
	}
	di := vu.View.Di
	curModeDir := ss.Current.Dir(mode.String())
	if curModeDir.Node("TrialName") == nil {
		return counters
	}
	counters += fmt.Sprintf(" TrialName: %s", curModeDir.StringValue("TrialName").String1D(di))
	statNames := []string{"Harmony"}
	if level == Cycle || curModeDir.Node(statNames[0]) == nil {
		return counters
	}
	for _, name := range statNames {
		counters += fmt.Sprintf(" %s: %.4g", name, curModeDir.Float64(name).Float1D(di))
	}
	return counters
}

//////// GUI

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	labs := []string{" Chloe Socks Sylv Garf Fuzz Daisy Fido Spot Snoop Penny",
		" black white brown orange", "bugs grass scraps shoe", "small  med  large", "cat     dog", "string feath bone shoe"}
	nv.ConfigLabels(labs)

	lays := []string{"Name", "Color", "FavoriteFood", "Size", "Species", "FavoriteToy"}

	for li, lnm := range lays {
		ly := nv.LayerByName(lnm)
		lbl := nv.LabelByName(labs[li])
		lbl.Pose = ly.Pose
		lbl.Pose.Pos.Y += .2
		lbl.Pose.Pos.Z += .02
		lbl.Pose.Scale.SetMul(math32.Vector3{0.4, 0.08, 0.5})
	}

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.5, 3.0)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc, readme)
	ss.GUI.StopLevel = Trial
	ss.GUI.CycleUpdateInterval = 1
	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 2 * 100
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.Net)
	ss.TestUpdate.Config(nv, leabra.Phase, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}

	ss.ConfigNetView(nv)

	ss.StatsInit()
	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	tree.Add(p, func(w *core.Separator) {})
}

func (ss *Sim) RunNoGUI() {
	ss.Init()

	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}

	runName := ss.SetRunName()
	netName := ss.Net.Name
	cfg := &ss.Config.Log
	leabra.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{cfg.Test})

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.Runs, ss.Config.Run.Run)
	ss.Loops.Loop(Train, Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.Runs)

	leabra.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
}
