// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// associator illustrates how error-driven and hebbian learning can
// operate within a simple task-driven learning context, with no hidden
// layers.
package associator

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
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/cluster"
	"cogentcore.org/lab/stats/metric"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed easy.tsv hard.tsv impossible.tsv
var embedfs embed.FS

//go:embed README.md
var readme embed.FS

// Patterns is the type of training patterns
type Patterns int32 //enums:enum

const (
	// Easy patterns can be learned by Hebbian learning
	Easy Patterns = iota

	// Hard patterns can only be learned with error-driven learning
	Hard

	// Impossible patterns require error-driven + a hidden layer
	Impossible
)

// LearnType is the type of learning to use
type LearnType int32 //enums:enum

const (
	Hebbian LearnType = iota
	ErrorDriven
)

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

	// select which type of learning to use
	Learn LearnType

	// select which type of patterns to use
	Patterns Patterns

	// simulation configuration parameters -- set by .toml config file and / or args
	Config *Config `new-window:"+"`

	// Net is the network: click to view / edit parameters for layers, paths, etc.
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// Params manages network parameter setting.
	Params leabra.Params `display:"inline"`

	// Loops are the control loops for running the sim, in different Modes
	// across stacks of Levels.
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// Envs provides mode-string based storage of environments.
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// TrainUpdate has Train mode netview update parameters.
	TrainUpdate leabra.NetViewUpdate `display:"inline"`

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
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.OpenInputs()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLoops()
	ss.ConfigStats()
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FixedTable{}
		tst = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(Train).(*env.FixedTable)
		tst = ss.Envs.ByMode(Test).(*env.FixedTable)
	}

	easy := tensorfs.DirTable(ss.Root.Dir("Inputs/Easy"), nil)

	// note: names must be standard here!
	trn.Name = Train.String()
	trn.Config(table.NewView(easy))
	trn.Validate()

	tst.Name = Test.String()
	tst.Config(table.NewView(easy))
	tst.Sequential = true
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	inp := net.AddLayer2D("Input", leabra.InputLayer, 1, 4)
	out := net.AddLayer2D("Output", leabra.TargetLayer, 1, 2)

	full := paths.NewFull()

	net.ConnectLayers(inp, out, full, leabra.ForwardPath)

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.ApplyAll(ss.Net)
	ss.Params.ApplySheet(ss.Net, ss.Learn.String())
	// if ss.Loops != nil {
	// 	trn := ss.Loops.Stacks[Train]
	// 	trn.Loops[Run].Counter.Max = ss.Config.Run.Runs
	// 	trn.Loops[Epoch].Counter.Max = ss.Config.Run.Epochs
	// }
}

////////  Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.SetRunName()
	ss.InitRandSeed(0)
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.ApplyParams()
	ss.StatsInit()
	ss.NewRun()
	ss.TrainUpdate.RecordSyns()
	ss.TrainUpdate.Update(Train, Trial)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// NetViewUpdater returns the NetViewUpdate for given mode.
func (ss *Sim) NetViewUpdater(mode enums.Enum) *leabra.NetViewUpdate {
	if mode.Int64() == Train.Int64() {
		return &ss.TrainUpdate
	}
	return &ss.TestUpdate
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trials := 4
	cycles := 100
	plusPhase := 25

	ls.AddStack(Train, Trial).
		AddLevel(Expt, 1).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevel(Trial, trials).
		AddLevel(Cycle, cycles)

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevel(Trial, trials).
		AddLevel(Cycle, cycles)

	leabra.LooperStandard(ls, ss.Net, ss.NetViewUpdater, cycles-plusPhase, cycles-1, Cycle, Trial, Train)

	ls.Stacks[Train].OnInit.Add("Init", ss.Init)

	ls.AddOnStartToLoop(Trial, "ApplyInputs", func(mode enums.Enum) {
		ss.ApplyInputs(mode.(Modes))
	})

	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	trainEpoch := ls.Loop(Train, Epoch)
	trainEpoch.IsDone.AddBool("NZeroStop", func() bool {
		stopNz := ss.Config.Run.NZero
		if stopNz <= 0 {
			return false
		}
		curModeDir := ss.Current.Dir(Train.String())
		curNZero := int(curModeDir.Value("NZero").Float1D(-1))
		stop := curNZero >= stopNz
		return stop
	})

	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
			ss.TestAll()
		}
	})

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	if ss.Config.GUI {
		leabra.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

		ls.Stacks[Train].OnInit.Add("GUI-Init", ss.GUI.UpdateWindow)
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

	out := ss.Net.LayerByName("Output")
	if mode == Test {
		out.Params.Type = leabra.CompareLayer // don't clamp plus phase
	} else {
		out.Params.Type = leabra.TargetLayer
	}

	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
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

func (ss *Sim) UpdateEnv() {
	trn := ss.Envs.ByMode(Train).(*env.FixedTable)
	tst := ss.Envs.ByMode(Test).(*env.FixedTable)
	switch ss.Patterns {
	case Easy:
		easy := tensorfs.DirTable(ss.Root.Dir("Inputs/Easy"), nil)
		trn.Table = easy
		tst.Table = easy
	case Hard:
		hard := tensorfs.DirTable(ss.Root.Dir("Inputs/Hard"), nil)
		trn.Table = hard
		tst.Table = hard
	case Impossible:
		impossible := tensorfs.DirTable(ss.Root.Dir("Inputs/Impossible"), nil)
		trn.Table = impossible
		tst.Table = impossible
	}
}

// NewRun intializes a new Run level of the model.
func (ss *Sim) NewRun() {
	ctx := ss.Net.Context()
	ss.InitRandSeed(ss.Loops.Loop(Train, Run).Counter.Cur)
	ss.UpdateEnv()
	ss.Envs.ByMode(Train).Init(0)
	ss.Envs.ByMode(Test).Init(0)
	ctx.Reset()
	ss.Net.InitWeights()
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(Test).Init(0)
	ss.Loops.ResetAndRun(Test)
	ss.Loops.Mode = Train // important because this is called from Train Run: go back.
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

func (ss *Sim) OpenInputs() {
	dir := ss.Root.Dir("Inputs")
	ss.OpenTable(dir, embedfs, "easy.tsv", "Easy", "Easy patterns")
	ss.OpenTable(dir, embedfs, "hard.tsv", "Hard", "Hard patterns")
	ss.OpenTable(dir, embedfs, "impossible.tsv", "Impossible", "Impossible patterns")
}

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
	if level <= Trial {
		return
	}
	ss.RunStats(mode, level-1, Start)
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level == Cycle {
		return
	}
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
			ss.GUI.Tabs.AsLab().GoUpdatePlot("Train RunAll Plot")
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
			if level == Cycle {
				continue
			}
			ss.RunStats(mode, level, Start)
		}
	}
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(leabra.StatsNode(ss.Stats, Train, Epoch))
		tbs.PlotTensorFS(leabra.StatsNode(ss.Stats, Train, Run))
		tbs.PlotTensorFS(leabra.StatsNode(ss.Stats, Test, Trial))
		tbs.PlotTensorFS(ss.Stats.Dir("Train/RunAll"))
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
	counterFunc := leabra.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		counterFunc(mode, level, phase == Start)
	})
	runNameFunc := leabra.StatRunName(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		runNameFunc(mode, level, phase == Start)
	})
	trialNameFunc := leabra.StatTrialName(ss.Stats, ss.Current, ss.Loops, net, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trialNameFunc(mode, level, phase == Start)
	})

	// up to a point, it is good to use loops over stats in one function,
	// to reduce repetition of boilerplate.
	statNames := []string{"CorSim", "SSE", "AvgSSE", "Err", "NZero", "FirstZero", "LastZero"}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		for _, name := range statNames {
			if name == "NZero" && (mode != Train || level == Trial) {
				return
			}
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
					case "SSE":
						s.On = true
					case "FirstZero", "LastZero":
						if level >= Run {
							s.On = true
						}
					}
				})
				switch name {
				case "NZero":
					if level == Epoch {
						curModeDir.Float64(name, 1).SetFloat1D(0, 0)
					}
				case "FirstZero", "LastZero":
					if level == Epoch {
						curModeDir.Float64(name, 1).SetFloat1D(-1, 0)
					}
				}
				continue
			}
			switch level {
			case Trial:
				out := ss.Net.LayerByName("Output")
				var stat float64
				switch name {
				case "CorSim":
					stat = 1.0 - float64(out.CosDiff.Cos)
				// case "UnitErr":
				// 	stat = out.PctUnitErr(ss.Net.Context())[0]
				case "SSE":
					sse, avgsse := out.MSE(0.5) // 0.5 = per-unit tolerance
					stat = sse
					curModeDir.Float64("AvgSSE", ndata).SetFloat1D(avgsse, 0)
				case "AvgSSE":
					stat = curModeDir.Float64("AvgSSE", ndata).Float1D(0)
				case "Err":
					uniterr := curModeDir.Float64("SSE", ndata).Float1D(0)
					stat = 1.0
					if uniterr == 0 {
						stat = 0
					}
				}
				curModeDir.Float64(name, ndata).SetFloat1D(stat, 0)
				tsr.AppendRowFloat(stat)
			case Epoch:
				nz := curModeDir.Float64("NZero", 1).Float1D(0)
				switch name {
				case "NZero":
					err := stats.StatSum.Call(subDir.Value("Err")).Float1D(0)
					stat = curModeDir.Float64(name, 1).Float1D(0)
					if err == 0 {
						stat++
					} else {
						stat = 0
					}
					curModeDir.Float64(name, 1).SetFloat1D(stat, 0)
				case "FirstZero":
					stat = curModeDir.Float64(name, 1).Float1D(0)
					if stat < 0 && nz == 1 {
						stat = curModeDir.Int("Epoch", 1).Float1D(0)
					}
					curModeDir.Float64(name, 1).SetFloat1D(stat, 0)
				case "LastZero":
					stat = curModeDir.Float64(name, 1).Float1D(0)
					if stat < 0 && nz >= float64(ss.Config.Run.NZero) {
						stat = curModeDir.Int("Epoch", 1).Float1D(0)
					}
					curModeDir.Float64(name, 1).SetFloat1D(stat, 0)
				default:
					stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				}
				tsr.AppendRowFloat(stat)
			case Run:
				stat = stats.StatFinal.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			default: // Expt
				stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			}
		}
	})

	perTrlFunc := leabra.StatPerTrialMSec(ss.Stats, Train, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		perTrlFunc(mode, level, phase == Start)
	})

	lays := net.LayersByType(leabra.SuperLayer, leabra.CTLayer, leabra.TargetLayer)
	actGeFunc := leabra.StatLayerActGe(ss.Stats, net, Train, Trial, Run, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		actGeFunc(mode, level, phase == Start)
	})

	stateFunc := leabra.StatLayerState(ss.Stats, net, Test, Trial, true, "ActM", "Input", "Output")
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		stateFunc(mode, level, phase == Start)
	})

	runAllFunc := leabra.StatLevelAll(ss.Stats, Train, Run, func(s *plot.Style, cl tensor.Values) {
		name := metadata.Name(cl)
		switch name {
		case "FirstZero", "LastZero":
			s.On = true
			s.Range.SetMin(0)
		}
	})
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		runAllFunc(mode, level, phase == Start)
	})
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
	statNames := []string{"CorSim", "SSE", "Err"}
	if level == Cycle || curModeDir.Node(statNames[0]) == nil {
		return counters
	}
	for _, name := range statNames {
		counters += fmt.Sprintf(" %s: %.4g", name, curModeDir.Float64(name).Float1D(di))
	}
	return counters
}

// ClusterPlot generates a cluster plot of input data.
func (ss *Sim) ClusterPlot() {
	plt := ss.GUI.Tabs.NewPlot("ClusterPlot")
	trn := ss.Envs.ByMode(Train).(*env.FixedTable)
	cluster.PlotFromTable(plt, trn.Table, metric.MetricL2Norm, cluster.Min, "Input", "Name")
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc, readme)
	ss.GUI.StopLevel = Trial
	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 2 * 100
	nv.Options.Raster.Max = 100
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, leabra.Phase, ss.StatCounters)
	ss.TestUpdate.Config(nv, leabra.Phase, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1, 2.75)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.Tabs.NewPlot("ClusterPlot")

	ss.GUI.Tabs.SelectTabIndex(0)
	ss.StatsInit()
	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "Cluster Plot",
		Icon:    icons.BarChart,
		Tooltip: "Generate a cluster plot of the input patterns.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.ClusterPlot()
		},
	})
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL(ss.Config.URL)
		},
	})
}

func (ss *Sim) RunNoGUI() {
	ss.Init()

	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}

	runName := ss.SetRunName()
	netName := ss.Net.Name
	cfg := &ss.Config.Log
	leabra.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{cfg.Train, cfg.Test})

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.Runs, ss.Config.Run.Run)
	ss.Loops.Loop(Train, Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.Runs)

	ss.Loops.Run(Train)

	leabra.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
}
