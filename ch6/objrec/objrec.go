// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
objrec explores how a hierarchy of areas in the ventral stream of visual
processing (up to inferotemporal (IT) cortex) can produce robust object
recognition that is invariant to changes in position, size, etc of retinal
input images.
*/
package main

//go:generate core generate -add-types

import (
	"embed"
	"fmt"
	"os"
	"reflect"

	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/system"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/etensor/tensor/stats/split"
	"github.com/emer/etensor/tensor/stats/stats"
	"github.com/emer/etensor/tensor/table"
	"github.com/emer/etensor/tensor/tensorcore"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed objrec_train1.wts.gz
var content embed.FS

//go:embed *.png *.md
var readme embed.FS

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		sim.RunGUI()
	} else {
		sim.RunNoGUI()
	}
}

// see params.go for params, config.go for Config

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	// Probability of training on novel items (0 for first phase, then .5 = 50%)
	PNovel float64

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config `new-window:"+"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// all parameter management
	Params emer.NetParams `display:"add-fields"`

	// contains looper control loops for running sim
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats `new-window:"+"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `new-window:"+"`

	// Environments
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// leabra timing parameters and state
	Context leabra.Context `new-window:"+"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"add-fields"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Config.Defaults()
	econfig.Config(&ss.Config, "config.toml")
	ss.Net = leabra.NewNetwork("Objrec")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Params.Params, &ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, novTrn, tst *LEDEnv
	if len(ss.Envs) == 0 {
		trn = &LEDEnv{}
		novTrn = &LEDEnv{}
		tst = &LEDEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*LEDEnv)
		novTrn = ss.Envs.ByMode(etime.Analyze).(*LEDEnv)
		tst = ss.Envs.ByMode(etime.Test).(*LEDEnv)
	}

	trn.Name = etime.Train.String()
	trn.Defaults()
	trn.MinLED = 0
	trn.MaxLED = 22 // exclude last 2 novel items by default
	if ss.Config.Env.Env != nil {
		params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
	}
	trn.Trial.Max = ss.Config.Run.NTrials

	novTrn.Name = etime.Analyze.String()
	novTrn.Defaults()
	novTrn.MinLED = 23
	novTrn.MaxLED = 24 // only last 2 novel items
	if ss.Config.Env.Env != nil {
		params.ApplyMap(novTrn, ss.Config.Env.Env, ss.Config.Debug)
	}
	novTrn.Trial.Max = ss.Config.Run.NTrials
	novTrn.XFormRand.TransX.Set(-0.125, 0.125)
	novTrn.XFormRand.TransY.Set(-0.125, 0.125)
	novTrn.XFormRand.Scale.Set(0.775, 0.925) // 1/2 around midpoint
	novTrn.XFormRand.Rot.Set(-2, 2)

	tst.Name = etime.Test.String()
	tst.Defaults()
	tst.MinLED = 0
	tst.MaxLED = 24      // all 25 patterns by default
	tst.Trial.Max = 500  // 0 // 1000 is too long!
	if ss.Config.Env.Env != nil {
		params.ApplyMap(tst, ss.Config.Env.Env, ss.Config.Debug)
	}

	trn.Init(0)
	novTrn.Init(0)
	tst.Init(0)

	ss.Envs.Add(trn, novTrn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	v1 := net.AddLayer4D("V1", 10, 10, 5, 4, leabra.InputLayer)
	v4 := net.AddLayer4D("V4", 5, 5, 7, 7, leabra.SuperLayer)
	it := net.AddLayer2D("IT", 10, 10, leabra.SuperLayer)
	out := net.AddLayer2D("Output", 5, 5, leabra.TargetLayer)

	v1.SetSampleIndexesShape(emer.CenterPoolIndexes(v1, 2), emer.CenterPoolShape(v1, 2))
	v4.SetSampleIndexesShape(emer.CenterPoolIndexes(v4, 2), emer.CenterPoolShape(v4, 2))

	full := paths.NewFull()

	net.ConnectLayers(v1, v4, ss.Config.Params.V1V4Path, leabra.ForwardPath)
	v4IT, _ := net.BidirConnectLayers(v4, it, full)
	itOut, outIT := net.BidirConnectLayers(it, out, full)

	it.PlaceRightOf(v4, 2)
	out.PlaceRightOf(it, 2)

	v4IT.AddClass("NovLearn")
	itOut.AddClass("NovLearn")
	outIT.AddClass("NovLearn")

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll() // first hard-coded defaults
	if ss.Config.Params.Network != nil {
		ss.Params.SetNetworkMap(ss.Net, ss.Config.Params.Network)
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	if ss.Config.GUI {
		ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	}
	ss.Loops.ResetCounters()
	ss.InitRandSeed(0)
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.NewRun()
	ss.ViewUpdate.RecordSyns()
	ss.ViewUpdate.Update()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trls := ss.Config.Run.NTrials

	ls.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, 500). // enough to get a better sample for actrf
		AddTime(etime.Cycle, 100)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 75, 99)                // plus phase timing
	leabra.LooperSimCycleAndLearn(ls, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	ls.Stacks[etime.Train].OnInit.Add("Init", func() { ss.Init() })

	for m := range ls.Stacks {
		stack := ls.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	ls.Loop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Add Testing
	trainEpoch := ls.Loop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

	/////////////////////////////////////////////
	// Logging

	ls.Loop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		leabra.LogTestErrors(&ss.Logs)
	})
	ls.Loop(etime.Train, etime.Epoch).OnEnd.Add("PCAStats", func() {
		trnEpc := ls.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if ss.Config.Run.PCAInterval > 0 && trnEpc%ss.Config.Run.PCAInterval == 0 {
			leabra.PCAStats(ss.Net, &ss.Logs, &ss.Stats)
			ss.Logs.ResetLog(etime.Analyze, etime.Trial)
		}
	})

	ls.AddOnEndToAll("Log", func(mode, time enums.Enum) {
		ss.Log(mode.(etime.Modes), time.(etime.Times))
	})
	leabra.LooperResetLogBelow(ls, &ss.Logs)

	ls.Loop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
		trnEpc := ls.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	ls.Loop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	// Save weights to file, to look at later
	ls.Loop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintValues([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		leabra.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.Stats.String("RunName"))
	})

	////////////////////////////////////////////
	// GUI

	if !ss.Config.GUI {
		if ss.Config.Log.NetData {
			ls.Loop(etime.Test, etime.Trial).OnEnd.Add("NetDataRecord", func() {
				ss.GUI.NetDataRecord(ss.ViewUpdate.Text)
			})
		}
	} else {
		ls.Loop(etime.Test, etime.Trial).OnEnd.Add("ActRFs", func() {
			ss.Stats.UpdateActRFs(ss.Net, "ActM", 0.01, 0)
		})
		ls.Loop(etime.Train, etime.Trial).OnStart.Add("UpdateImage", func() {
			if system.TheApp.Platform() == system.Web { // todo: hangs on web
				return
			}
			ss.GUI.Grid("Image").NeedsRender()
		})
		ls.Loop(etime.Test, etime.Trial).OnStart.Add("UpdateImage", func() {
			if system.TheApp.Platform() == system.Web {
				return
			}
			ss.GUI.Grid("Image").NeedsRender()
		})

		leabra.LooperUpdateNetView(ls, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
		leabra.LooperUpdatePlots(ls, &ss.GUI)
		ls.Stacks[etime.Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
		ls.Stacks[etime.Test].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
	}

	if ss.Config.Debug {
		fmt.Println(ls.DocString())
	}
	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*LEDEnv)
	if ctx.Mode == etime.Train && randx.BoolP(ss.PNovel) {
		ev = ss.Envs.ByMode(etime.Analyze).(*LEDEnv)
	}
	net.InitExt()
	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	ev.Step()

	ss.Stats.SetInt("Cat", ev.CurLED) // note: must save relevant state for stats later
	ss.Stats.SetString("TrialName", ev.String())
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm)
		pats := ev.State(ly.Name)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(ss.Loops.Loop(etime.Train, etime.Run).Counter.Cur)
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWeights()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Stats.ActRFs.Reset()
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
	ss.Stats.ActRFsAvgNorm()
	ss.GUI.ViewActRFs(&ss.Stats.ActRFs)

}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.Logs.ResetLog(etime.Test, etime.Epoch) // only show last row
	ss.GUI.StopNow = false
	ss.TestAll()
	ss.GUI.Stopped()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("SSE", 0.0)
	ss.Stats.SetInt("Cat", 0)
	ss.Stats.SetString("Cat", "0")
	ss.Stats.SetString("TrialName", "0")
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdate.Text
func (ss *Sim) StatCounters() {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CountersToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
	ss.Stats.SetString("TrialName", ss.Stats.String("TrialName"))
	ss.Stats.SetString("Cat", fmt.Sprintf("%02d", ss.Stats.Int("Cat")))
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	if tm == etime.Trial {
		ss.TrialStats() // get trial stats for current di
	}
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Cat", "TrialName", "Cycle", "SSE", "TrlErr"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	ctx := &ss.Context
	out := ss.Net.LayerByName("Output")
	sse, avgsse := out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	ss.Stats.SetFloat("SSE", sse)
	ss.Stats.SetFloat("AvgSSE", avgsse)
	ev := ss.Envs.ByMode(ctx.Mode).(*LEDEnv)
	ovt := ss.Stats.SetLayerTensor(ss.Net, "Output", "ActM", 0)
	cat := ss.Stats.Int("Cat")
	rsp, trlErr, trlErr2 := ev.OutErr(ovt, cat)
	ss.Stats.SetFloat("TrlErr", trlErr)
	ss.Stats.SetFloat("TrlErr2", trlErr2)
	ss.Stats.SetString("TrlOut", fmt.Sprintf("%d", rsp))
	ss.Stats.SetString("Cat", fmt.Sprintf("%d", cat))
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "Cat", "TrialName")

	ss.Logs.AddStatAggItem("SSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("AvgSSE", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	ss.Logs.AddCopyFromFloatItems(etime.Train, []etime.Times{etime.Epoch, etime.Run}, etime.Test, etime.Epoch, "Tst", "SSE", "PctCor", "PctErr")

	ss.ConfigActRFs()

	layers := ss.Net.LayersByType(leabra.SuperLayer, leabra.CTLayer, leabra.TargetLayer)
	leabra.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	leabra.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	leabra.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "TargetLayer")

	// this was useful during development of trace learning:
	// leabra.LogAddCaLrnDiagnosticItems(&ss.Logs, ss.Net, etime.Epoch, etime.Trial)

	ss.Logs.PlotItems("PctErr")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "Type", "Bar")
}

// ConfigLogItems specifies extra logging items
func (ss *Sim) ConfigLogItems() {
	ss.Logs.AddItem(&elog.Item{
		Name: "Err2",
		Type: reflect.Float64,
		Plot: true,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlErr2")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "PctErr2",
		Type: reflect.Float64,
		Plot: false,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAggItem(ctx.Mode, etime.Trial, "Err2", stats.Mean)
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
				ctx.SetFloat64(stats.MeanColumn(ix, ctx.Item.Name)[0])
			}}})

	ss.Logs.AddItem(&elog.Item{
		Name:        "CatErr",
		Type:        reflect.Float64,
		CellShape:   []int{25},
		DimNames:    []string{"Cat"},
		Plot:        true,
		Range:       minmax.F32{Min: 0},
		TensorIndex: -1, // plot all values
		Write: elog.WriteMap{
			etime.Scope(etime.Test, etime.Epoch): func(ctx *elog.Context) {
				ix := ctx.Logs.IndexView(etime.Test, etime.Trial)
				spl := split.GroupBy(ix, "Cat")
				split.AggColumn(spl, "Err", stats.Mean)
				cats := spl.AggsToTable(table.ColumnNameOnly)
				ss.Logs.MiscTables[ctx.Item.Name] = cats
				ctx.SetTensor(cats.Columns[1])
			}}})
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
		ss.Context.Mode = mode // Also set specifically in a Loop callback.
	}
	dt := ss.Logs.Table(mode, time)
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return
	case time == etime.Trial:
		ss.TrialStats()
		ss.StatCounters()
		ss.Logs.LogRow(mode, time, row)
		return // don't do reg below
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

// ConfigActRFs
func (ss *Sim) ConfigActRFs() {
	ss.Stats.SetF32Tensor("Image", &ss.Envs.ByMode(etime.Test).(*LEDEnv).Vis.ImgTsr) // image used for actrfs, must be there first
	ss.Stats.InitActRFs(ss.Net, []string{"V4:Image", "V4:Output", "IT:Image", "IT:Output"}, "ActM")
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Object Recognition"
	ss.GUI.MakeBody(ss, "objrec", title, `This simulation explores how a hierarchy of areas in the ventral stream of visual processing (up to inferotemporal (IT) cortex) can produce robust object recognition that is invariant to changes in position, size, etc of retinal input images. See <a href="https://github.com/compcogneuro/sims/blob/main/ch6/objrec/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.Options.Raster.Max = 100
	nv.Options.LayerNameSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.GammaCycle, etime.GammaCycle)

	cam := &(nv.SceneXYZ().Camera)
	cam.Pose.Pos.Set(0.0, 1.733, 2.3)
	cam.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	itb, _ := ss.GUI.Tabs.NewTab("Image")
	tg := tensorcore.NewTensorGrid(itb).
		SetTensor(&ss.Envs.ByMode(etime.Train).(*LEDEnv).Vis.ImgTsr)
	ss.GUI.SetGrid("Image", tg)

	ss.GUI.AddActRFGridTabs(&ss.Stats.ActRFs)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Test All",
		Icon:    icons.PlayArrow,
		Tooltip: "Tests a large same of testing items and records ActRFs.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.UpdateWindow()
				go ss.RunTestAll()
			}
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Open Trained Wts", Icon: icons.Open,
		Tooltip: "Opened weights from the first phase of training, which excludes novel objects",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Net.OpenWeightsFS(content, "objrec_train1.wts.gz")
			ss.ViewUpdate.RecordSyns()
			ss.ViewUpdate.Update()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Train Novel", Icon: icons.Update,
		Tooltip: "Configure for training novel objects: loads phase 1 weights and sets PNovel = .5, etc",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.NewRun()
			ss.Params.SetAllSheet("NovelLearn")
			ss.PNovel = 0.5
			ss.GUI.SimForm.Update()
			ss.GUI.UpdateWindow()
			go func() {
				ss.Loops.Step(etime.Train, 1, etime.Trial)
				ss.Net.OpenWeightsFS(content, "objrec_train1.wts.gz")
			}()
		},
	})

	////////////////////////////////////////////////
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    icons.Reset,
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/compcogneuro/sims/blob/main/ch6/objrec/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}

func (ss *Sim) RunNoGUI() {
	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWeights {
		mpi.Printf("Saving final weights per run\n")
	}
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name

	elog.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Train, etime.Trial, "trl", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Train, etime.Epoch, "epc", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.TestEpoch, etime.Test, etime.Epoch, "tst_epc", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.TestTrial, etime.Test, etime.Trial, "tst_trl", netName, runName)

	netdata := ss.Config.Log.NetData
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	ss.Init()

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.NRuns, ss.Config.Run.Run)
	ss.Loops.Loop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.NRuns)

	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}
}
