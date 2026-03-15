// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// hip runs a hippocampus model on the AB-AC paired associate learning task.
package main

//go:generate core generate -add-types

import (
	"embed"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tree"
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
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/etensor/plot/plotcore"
	"github.com/emer/etensor/tensor/stats/split"
	"github.com/emer/etensor/tensor/table"
	"github.com/emer/leabra/v2/leabra"
)

//go:embed train_ab.tsv train_ac.tsv test_ab.tsv test_ac.tsv test_lure.tsv
var content embed.FS

//go:embed *.png README.md
var readme embed.FS

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	sim.RunGUI()
}

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Path", Desc: "keeping default params for generic prjns",
			Params: params.Params{
				"Path.Learn.Momentum.On": "true",
				"Path.Learn.Norm.On":     "true",
				"Path.Learn.WtBal.On":    "false",
			}},
		{Sel: ".EcCa1Path", Desc: "encoder projections -- no norm, moment",
			Params: params.Params{
				"Path.Learn.Lrate":        "0.04",
				"Path.Learn.Momentum.On":  "false",
				"Path.Learn.Norm.On":      "false",
				"Path.Learn.WtBal.On":     "true",
				"Path.Learn.XCal.SetLLrn": "false", // using bcm now, better
			}},
		{Sel: ".HippoCHL", Desc: "hippo CHL projections -- no norm, moment, but YES wtbal = sig better",
			Params: params.Params{
				"Path.CHL.Hebb":          "0.05",
				"Path.Learn.Lrate":       "0.2",
				"Path.Learn.Momentum.On": "false",
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.WtBal.On":    "true",
			}},
		{Sel: ".PPath", Desc: "perforant path, new Dg error-driven EcCa1Path prjns",
			Params: params.Params{
				"Path.Learn.Momentum.On": "false",
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.WtBal.On":    "true",
				"Path.Learn.Lrate":       "0.15", // err driven: .15 > .2 > .25 > .1
				// moss=4, delta=4, lr=0.2, test = 3 are best
			}},
		{Sel: "#CA1ToECout", Desc: "extra strong from CA1 to ECout",
			Params: params.Params{
				"Path.WtScale.Abs": "4.0",
			}},
		{Sel: "#InputToECin", Desc: "one-to-one input to EC",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "0.8",
				"Path.WtInit.Var":  "0.0",
			}},
		{Sel: "#ECoutToECin", Desc: "one-to-one out to in",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "0.9",
				"Path.WtInit.Var":  "0.01",
				"Path.WtScale.Rel": "0.5",
			}},
		{Sel: "#DGToCA3", Desc: "Mossy fibers: strong, non-learning",
			Params: params.Params{
				"Path.Learn.Learn": "false",
				"Path.WtInit.Mean": "0.9",
				"Path.WtInit.Var":  "0.01",
				"Path.WtScale.Rel": "4",
			}},
		{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons",
			Params: params.Params{
				"Path.WtScale.Rel": "0.1",
				"Path.Learn.Lrate": "0.1",
			}},
		{Sel: "#ECinToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
			Params: params.Params{
				"Path.Learn.Learn":       "true", // absolutely essential to have on!
				"Path.CHL.Hebb":          ".5",   // .5 > 1 overall
				"Path.CHL.SAvgCor":       "0.1",  // .1 > .2 > .3 > .4 ?
				"Path.CHL.MinusQ1":       "true", // dg self err?
				"Path.Learn.Lrate":       "0.4",  // .4 > .3 > .2
				"Path.Learn.Momentum.On": "false",
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.WtBal.On":    "true",
			}},
		{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
			Params: params.Params{
				"Path.CHL.Hebb":          "0.01",
				"Path.CHL.SAvgCor":       "0.4",
				"Path.Learn.Lrate":       "0.1",
				"Path.Learn.Momentum.On": "false",
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.WtBal.On":    "true",
			}},
		{Sel: ".EC", Desc: "all EC layers: only pools, no layer-level",
			Params: params.Params{
				"Layer.Act.Gbar.L":        ".1",
				"Layer.Inhib.ActAvg.Init": "0.2",
				"Layer.Inhib.Layer.On":    "false",
				"Layer.Inhib.Pool.Gi":     "2.0",
				"Layer.Inhib.Pool.On":     "true",
			}},
		{Sel: "#DG", Desc: "very sparse = high inibhition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.01",
				"Layer.Inhib.Layer.Gi":    "3.8",
			}},
		{Sel: "#CA3", Desc: "sparse = high inibhition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.02",
				"Layer.Inhib.Layer.Gi":    "2.8",
			}},
		{Sel: "#CA1", Desc: "CA1 only Pools",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Init": "0.1",
				"Layer.Inhib.Layer.On":    "false",
				"Layer.Inhib.Pool.Gi":     "2.4",
				"Layer.Inhib.Pool.On":     "true",
			}},
	},
}

// Config has config parameters related to running the sim
type Config struct {
	// total number of runs to do when running Train
	NRuns int `default:"10" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"20"`

	// stop run after this number of perfect, zero-error epochs.
	NZero int `default:"1"`

	// how often to run through all the test patterns, in terms of training epochs.
	// can use 0 or -1 for no testing.
	TestInterval int `default:"1"`

	// StopMem is the threshold for stopping learning.
	StopMem float32 `default:"1"`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

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

	// if true, run in pretrain mode
	PretrainMode bool `display:"-"`

	// DGLesion controls whether a 50% lesion is applied to the DG (dentate gyrus) layer
	// before AC list training begins. The lesion is automatically removed at the start of
	// each new Run so that AB training is always unaffected.
	DGLesion bool

	// pool patterns vocabulary
	PoolVocab patgen.Vocab `display:"-"`

	// AB training patterns to use
	TrainAB *table.Table `new-window:"+" display:"no-inline"`

	// AC training patterns to use
	TrainAC *table.Table `new-window:"+" display:"no-inline"`

	// AB testing patterns to use
	TestAB *table.Table `new-window:"+" display:"no-inline"`

	// AC testing patterns to use
	TestAC *table.Table `new-window:"+" display:"no-inline"`

	// Lure testing patterns to use
	TestLure *table.Table `new-window:"+" display:"no-inline"`

	// TestAll has all the test items
	TestAll *table.Table `new-window:"+" display:"no-inline"`

	// Lure pretrain patterns to use
	PreTrainLure *table.Table `new-window:"+" display:"-"`

	// all training patterns -- for pretrain
	TrainAll *table.Table `new-window:"+" display:"-"`

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
	// ss.Config.Defaults()
	econfig.Config(&ss.Config, "config.toml")
	// ss.Config.Hip.EC5Clamp = true      // must be true in hip.go to have a target layer
	// ss.Config.Hip.EC5ClampTest = false // key to be off for cmp stats on completion region

	ss.Net = leabra.NewNetwork("Hip")
	ss.Params.Config(ParamSets, "", "", ss.Net)
	ss.Stats.Init()
	ss.Stats.SetInt("Expt", 0)

	ss.PoolVocab = patgen.Vocab{}
	ss.TrainAB = &table.Table{}
	ss.TrainAC = &table.Table{}
	ss.TestAB = &table.Table{}
	ss.TestAC = &table.Table{}
	ss.PreTrainLure = &table.Table{}
	ss.TestLure = &table.Table{}
	ss.TrainAll = &table.Table{}
	ss.TestAll = &table.Table{}
	ss.PretrainMode = false

	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.OpenPatterns()
	// ss.ConfigPatterns()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FixedTable{}
		tst = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*env.FixedTable)
		tst = ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	}

	// note: names must be standard here!
	trn.Name = etime.Train.String()
	trn.Config(table.NewIndexView(ss.TrainAB))
	trn.Validate()

	tst.Name = etime.Test.String()
	tst.Config(table.NewIndexView(ss.TestAll))
	tst.Sequential = true
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	in := net.AddLayer4D("Input", 6, 2, 3, 4, leabra.InputLayer)
	ecin := net.AddLayer4D("ECin", 6, 2, 3, 4, leabra.SuperLayer)
	ecout := net.AddLayer4D("ECout", 6, 2, 3, 4, leabra.TargetLayer) // clamped in plus phase
	ca1 := net.AddLayer4D("CA1", 6, 2, 4, 10, leabra.SuperLayer)
	dg := net.AddLayer2D("DG", 25, 25, leabra.SuperLayer)
	ca3 := net.AddLayer2D("CA3", 30, 10, leabra.SuperLayer)

	ecin.AddClass("EC")
	ecout.AddClass("EC")

	onetoone := paths.NewOneToOne()
	pool1to1 := paths.NewPoolOneToOne()
	full := paths.NewFull()

	net.ConnectLayers(in, ecin, onetoone, leabra.ForwardPath)
	net.ConnectLayers(ecout, ecin, onetoone, leabra.BackPath)

	// EC <-> CA1 encoder pathways
	net.ConnectLayers(ecin, ca1, pool1to1, leabra.EcCa1Path)
	net.ConnectLayers(ca1, ecout, pool1to1, leabra.EcCa1Path)
	net.ConnectLayers(ecout, ca1, pool1to1, leabra.EcCa1Path)

	// Perforant pathway
	ppath := paths.NewUniformRand()
	ppath.PCon = 0.25

	net.ConnectLayers(ecin, dg, ppath, leabra.CHLPath).AddClass("HippoCHL")

	net.ConnectLayers(ecin, ca3, ppath, leabra.EcCa1Path).AddClass("PPath")
	net.ConnectLayers(ca3, ca3, full, leabra.EcCa1Path).AddClass("PPath")

	// Mossy fibers
	mossy := paths.NewUniformRand()
	mossy.PCon = 0.02
	net.ConnectLayers(dg, ca3, mossy, leabra.CHLPath).AddClass("HippoCHL")

	// Schafer collaterals
	net.ConnectLayers(ca3, ca1, full, leabra.CHLPath).AddClass("HippoCHL")

	ecin.PlaceRightOf(in, 2)
	ecout.PlaceRightOf(ecin, 2)
	dg.PlaceAbove(in)
	ca3.PlaceAbove(dg)
	ca1.PlaceRightOf(ca3, 2)

	in.Doc = "Input represents cortical processing areas for different sensory modalities, semantic categories, etc, organized into pools. It is pre-compressed in this model, to simplify and allow one-to-one projections into the EC."

	ecin.Doc = "Entorhinal Cortex (EC) input layer is the superficial layer 2 that receives from the cortex and projects into the hippocampus. It has compressed representations of cortical inputs."

	ecout.Doc = "Entorhinal Cortex (EC) output layer is the deep layers that are bidirectionally connected to the CA1, and communicate hippocampal recall back out to the cortex, while also training the CA1 to accurately represent the EC inputs"

	ca1.Doc = "CA (Cornu Ammonis = Ammon's horn) area 1, receives from CA3 and drives recalled memory output to ECout"

	ca3.Doc = "CA (Cornu Ammonis = Ammon's horn) area 3, receives inputs from ECin and DG, and is the primary site of memory encoding. Recurrent self-connections drive pattern completion of full memory representations from partial cues, along with connections to CA1 that drive memory output."

	dg.Doc = "Dentate Gyruns, which receives broad inputs from ECin and has highly sparse, pattern separated representations, which drive more separated representations in CA3"

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
	net.InitTopoScales()
}

func (ss *Sim) ApplyParams() {
	ss.Params.Network = ss.Net
	ss.Params.SetAll()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	ss.Loops.ResetCounters()

	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.NewRun()
	ss.ViewUpdate.RecordSyns()
	ss.ViewUpdate.Update()
}

func (ss *Sim) TestInit() {
	tst := ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	tst.Init(0)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	rand.Seed(ss.RandSeeds[run])
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
	patgen.NewRand(ss.RandSeeds[run])
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trls := ss.TrainAB.Rows
	ttrls := ss.TestAll.Rows

	ls.AddStack(etime.Train).AddTime(etime.Run, ss.Config.NRuns).AddTime(etime.Epoch, ss.Config.NEpochs).AddTime(etime.Trial, trls).AddTime(etime.Cycle, 100)

	ls.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTime(etime.Trial, ttrls).AddTime(etime.Cycle, 100)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 75, 99)                // plus phase timing
	leabra.LooperSimCycleAndLearn(ls, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code
	ss.Net.ConfigLoopsHip(&ss.Context, ls)

	ls.Stacks[etime.Train].OnInit.Add("Init", func() { ss.Init() })
	ls.Stacks[etime.Test].OnInit.Add("Init", func() { ss.TestInit() })

	for m, _ := range ls.Stacks {
		stack := ls.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	ls.Loop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	ls.Loop(etime.Train, etime.Run).OnEnd.Add("RunDone", func() {
		if ss.Stats.Int("Run") >= ss.Config.NRuns-1 {
			ss.RunStats()
			expt := ss.Stats.Int("Expt")
			ss.Stats.SetInt("Expt", expt+1)
		}
	})

	// Add Testing
	trainEpoch := ls.Loop(etime.Train, etime.Epoch)
	trainEpoch.OnEnd.Add("TestAtInterval", func() {
		if (ss.Config.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.RunTestAll()

			// switch to AC
			trn := ss.Envs.ByMode(etime.Train).(*env.FixedTable)
			tstEpcLog := ss.Logs.Tables[etime.Scope(etime.Test, etime.Epoch)]
			epc := ss.Stats.Int("Epoch")
			abMem := float32(tstEpcLog.Table.Float("ABMem", epc))
			if (trn.Table.Table.MetaData["name"] == "TrainAB") && (abMem >= ss.Config.StopMem || epc >= ss.Config.NEpochs/2) {
				ss.Stats.SetInt("FirstPerfect", epc)
				trn.Config(table.NewIndexView(ss.TrainAC))
				trn.Validate()
				// Apply DG lesion for AC training if the flag is set
				if ss.DGLesion {
					ss.Net.LayerByName("DG").LesionNeurons(0.5)
				}
			}
		}
	})

	// early stop
	ls.Loop(etime.Train, etime.Epoch).IsDone.AddBool("ACMemStop", func() bool {
		// This is calculated in TrialStats
		tstEpcLog := ss.Logs.Tables[etime.Scope(etime.Test, etime.Epoch)]
		acMem := float32(tstEpcLog.Table.Float("ACMem", ss.Stats.Int("Epoch")))
		stop := acMem >= ss.Config.StopMem
		return stop
	})

	/////////////////////////////////////////////
	// Logging

	ls.Loop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		leabra.LogTestErrors(&ss.Logs)
	})

	ls.AddOnEndToAll("Log", func(mode, time enums.Enum) {
		ss.Log(mode.(etime.Modes), time.(etime.Times))
	})
	leabra.LooperResetLogBelow(ls, &ss.Logs)

	leabra.LooperUpdateNetView(ls, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
	leabra.LooperUpdatePlots(ls, &ss.GUI)

	ls.Stacks[etime.Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
	ls.Stacks[etime.Test].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })

	ss.Loops = ls
	// fmt.Println(ls.DocString())
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*env.FixedTable)
	ecout := net.LayerByName("ECout")
	if ctx.Mode == etime.Train {
		ecout.Type = leabra.TargetLayer // clamp a plus phase during testing
	} else {
		ecout.Type = leabra.CompareLayer // don't clamp
	}
	ecout.UpdateExtFlags() // call this after updating type
	net.InitExt()
	lays := net.LayersByType(leabra.InputLayer, leabra.TargetLayer)
	ev.Step()
	// note: must save env state for logging / stats due to data parallel re-use of same env
	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
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
	// ss.ConfigPats()
	ss.ConfigEnv()
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWeights()
	// Restore DG to its full (un-lesioned) state for AB training
	ss.Net.LayerByName("DG").UnLesionNeurons()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) RunTestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

/////////////////////////////////////////////////////////////////////////
//   Pats

// OpenPatAsset opens pattern file from embedded assets
func (ss *Sim) OpenPatAsset(dt *table.Table, fnm, name, desc string) error {
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
	err := dt.OpenFS(content, fnm, table.Tab)
	if errors.Log(err) == nil {
		for i := 1; i < dt.NumColumns(); i++ {
			dt.Columns[i].SetMetaData("grid-fill", "0.9")
		}
	}
	return err
}

func (ss *Sim) OpenPatterns() {
	ss.OpenPatAsset(ss.TrainAB, "train_ab.tsv", "TrainAB", "AB Training Patterns")
	ss.OpenPatAsset(ss.TrainAC, "train_ac.tsv", "TrainAC", "AC Training Patterns")
	ss.OpenPatAsset(ss.TestAB, "test_ab.tsv", "TestAB", "AB Testing Patterns")
	ss.OpenPatAsset(ss.TestAC, "test_ac.tsv", "TestAC", "AC Testing Patterns")
	ss.OpenPatAsset(ss.TestLure, "test_lure.tsv", "TestLure", "Lure Testing Patterns")

	ss.TestAll = ss.TestAB.Clone()
	ss.TestAll.SetMetaData("name", "TestAll")
	ss.TestAll.AppendRows(ss.TestAC)
	ss.TestAll.AppendRows(ss.TestLure)
}

func (ss *Sim) ConfigPats() {
	// hp := &ss.Config.Hip
	ecY := 3               // hp.EC3NPool.Y
	ecX := 4               // hp.EC3NPool.X
	plY := 6               // hp.EC3NNrn.Y // good idea to get shorter vars when used frequently
	plX := 2               // hp.EC3NNrn.X // makes much more readable
	npats := 10            // ss.Config.NTrials
	pctAct := float32(.15) // ss.Config.Mod.ECPctAct
	minDiff := float32(.5) // ss.Config.Pat.MinDiffPct
	nOn := patgen.NFromPct(pctAct, plY*plX)
	ctxtFlipPct := float32(0.2)
	ctxtflip := patgen.NFromPct(ctxtFlipPct, nOn)
	patgen.AddVocabEmpty(ss.PoolVocab, "empty", npats, plY, plX)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "A", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "B", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "C", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt", 3, plY, plX, pctAct, minDiff) // totally diff

	for i := 0; i < (ecY-1)*ecX*3; i++ { // 12 contexts! 1: 1 row of stimuli pats; 3: 3 diff ctxt bases
		list := i / ((ecY - 1) * ecX)
		ctxtNm := fmt.Sprintf("ctxt%d", i+1)
		tsr, _ := patgen.AddVocabRepeat(ss.PoolVocab, ctxtNm, npats, "ctxt", list)
		patgen.FlipBitsRows(tsr, ctxtflip, ctxtflip, 1, 0)
		//todo: also support drifting
		//solution 2: drift based on last trial (will require sequential learning)
		//patgen.VocabDrift(ss.PoolVocab, ss.NFlipBits, "ctxt"+strconv.Itoa(i+1))
	}

	patgen.InitPats(ss.TrainAB, "TrainAB", "TrainAB Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainAB, ss.PoolVocab, "Input", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(ss.TrainAB, ss.PoolVocab, "ECout", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(ss.TestAB, "TestAB", "TestAB Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(ss.TestAB, ss.PoolVocab, "ECout", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(ss.TrainAC, "TrainAC", "TrainAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainAC, ss.PoolVocab, "Input", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(ss.TrainAC, ss.PoolVocab, "ECout", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(ss.TestAC, "TestAC", "TestAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestAC, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(ss.TestAC, ss.PoolVocab, "ECout", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(ss.PreTrainLure, "PreTrainLure", "PreTrainLure Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.PreTrainLure, ss.PoolVocab, "Input", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(ss.PreTrainLure, ss.PoolVocab, "ECout", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here

	patgen.InitPats(ss.TestLure, "TestLure", "TestLure Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestLure, ss.PoolVocab, "Input", []string{"lA", "empty", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(ss.TestLure, ss.PoolVocab, "ECout", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"})    // arbitrary ctxt here

	ss.TrainAll = ss.TrainAB.Clone()
	ss.TrainAll.AppendRows(ss.TrainAC)
	ss.TrainAll.AppendRows(ss.PreTrainLure)
	ss.TrainAll.MetaData["name"] = "TrainAll"
	ss.TrainAll.MetaData["desc"] = "All Training Patterns"

	ss.TestAll = ss.TestAB.Clone()
	ss.TestAll.AppendRows(ss.TestAC)
	ss.TestAll.MetaData["name"] = "TestAll"
	ss.TestAll.MetaData["desc"] = "All Testing Patterns"
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetString("TrialName", "")
	ss.Stats.SetFloat("TrgOnWasOffAll", 0.0)
	ss.Stats.SetFloat("TrgOnWasOffCmp", 0.0)
	ss.Stats.SetFloat("TrgOffWasOn", 0.0)
	ss.Stats.SetFloat("ABMem", 0.0)
	ss.Stats.SetFloat("ACMem", 0.0)
	ss.Stats.SetFloat("LureMem", 0.0)
	ss.Stats.SetFloat("Mem", 0.0)
	ss.Stats.SetInt("FirstPerfect", -1) // first epoch at when AB Mem is perfect

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
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	if tm == etime.Trial {
		ss.TrialStats() // get trial stats for current di
	}
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	ss.MemStats(ss.Loops.Mode.(etime.Modes))
}

// MemStats computes ActM vs. Target on ECout with binary counts
// must be called at end of 3rd quarter so that Target values are
// for the entire full pattern as opposed to the plus-phase target
// values clamped from ECin activations
func (ss *Sim) MemStats(mode etime.Modes) {
	memthr := 0.34 // ss.Config.Mod.MemThr
	ecout := ss.Net.LayerByName("ECout")
	inp := ss.Net.LayerByName("Input") // note: must be input b/c ECin can be active
	_ = inp
	nn := ecout.Shape.Len()
	actThr := float32(0.5)
	trgOnWasOffAll := 0.0 // all units
	trgOnWasOffCmp := 0.0 // only those that required completion, missing in ECin
	trgOffWasOn := 0.0    // should have been off
	cmpN := 0.0           // completion target
	trgOnN := 0.0
	trgOffN := 0.0
	actMi, _ := ecout.UnitVarIndex("ActM")
	targi, _ := ecout.UnitVarIndex("Targ")

	ss.Stats.SetFloat("ABMem", math.NaN())
	ss.Stats.SetFloat("ACMem", math.NaN())
	ss.Stats.SetFloat("LureMem", math.NaN())

	trialnm := ss.Stats.String("TrialName")
	isAB := strings.Contains(trialnm, "ab")
	isAC := strings.Contains(trialnm, "ac")

	for ni := 0; ni < nn; ni++ {
		actm := ecout.UnitValue1D(actMi, ni, 0)
		trg := ecout.UnitValue1D(targi, ni, 0) // full pattern target
		inact := inp.UnitValue1D(actMi, ni, 0)
		if trg < actThr { // trgOff
			trgOffN += 1
			if actm > actThr {
				trgOffWasOn += 1
			}
		} else { // trgOn
			trgOnN += 1
			if inact < actThr { // missing in ECin -- completion target
				cmpN += 1
				if actm < actThr {
					trgOnWasOffAll += 1
					trgOnWasOffCmp += 1
				}
			} else {
				if actm < actThr {
					trgOnWasOffAll += 1
				}
			}
		}
	}
	trgOnWasOffAll /= trgOnN
	trgOffWasOn /= trgOffN
	if mode == etime.Train { // no compare
		if trgOnWasOffAll < memthr && trgOffWasOn < memthr {
			ss.Stats.SetFloat("Mem", 1)
		} else {
			ss.Stats.SetFloat("Mem", 0)
		}
	} else { // test
		if cmpN > 0 { // should be
			trgOnWasOffCmp /= cmpN
		}
		mem := 0.0
		if trgOnWasOffCmp < memthr && trgOffWasOn < memthr {
			mem = 1.0
		}
		ss.Stats.SetFloat("Mem", mem)
		switch {
		case isAB:
			ss.Stats.SetFloat("ABMem", mem)
		case isAC:
			ss.Stats.SetFloat("ACMem", mem)
		default:
			ss.Stats.SetFloat("LureMem", mem)
		}

	}
	ss.Stats.SetFloat("TrgOnWasOffAll", trgOnWasOffAll)
	ss.Stats.SetFloat("TrgOnWasOffCmp", trgOnWasOffCmp)
	ss.Stats.SetFloat("TrgOffWasOn", trgOffWasOn)

}

func (ss *Sim) RunStats() {
	dt := ss.Logs.Table(etime.Train, etime.Run)
	runix := table.NewIndexView(dt)
	spl := split.GroupBy(runix, "Expt")
	split.DescColumn(spl, "TstABMem")
	st := spl.AggsToTableCopy(table.AddAggName)
	ss.Logs.MiscTables["RunStats"] = st
	plt := ss.GUI.Plots[etime.ScopeKey("RunStats")]

	st.SetMetaData("XAxis", "RunName")

	st.SetMetaData("Points", "true")

	st.SetMetaData("TstABMem:Mean:On", "+")
	st.SetMetaData("TstABMem:Mean:FixMin", "true")
	st.SetMetaData("TstABMem:Mean:FixMax", "true")
	st.SetMetaData("TstABMem:Mean:Min", "0")
	st.SetMetaData("TstABMem:Mean:Max", "1")
	st.SetMetaData("TstABMem:Min:On", "+")
	st.SetMetaData("TstABMem:Count:On", "-")

	plt.SetTable(st)
	plt.GoUpdatePlot()
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) AddLogItems() {
	itemNames := []string{"TrgOnWasOffAll", "TrgOnWasOffCmp", "TrgOffWasOn", "Mem", "ABMem", "ACMem", "LureMem"}
	for _, st := range itemNames {
		stnm := st
		tonm := "Tst" + st
		ss.Logs.AddItem(&elog.Item{
			Name: tonm,
			Type: reflect.Float64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetFloat64(ctx.ItemFloat(etime.Test, etime.Epoch, stnm))
				},
				etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
					ctx.SetFloat64(ctx.ItemFloat(etime.Test, etime.Epoch, stnm)) // take the last epoch
					// ctx.SetAgg(ctx.Mode, etime.Epoch, stats.Max) // stats.Max for max over epochs
				}}})
	}
}

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.AllTimes, "Expt")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddStatAggItem("TrgOnWasOffAll", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("TrgOnWasOffCmp", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("TrgOffWasOn", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ABMem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("ACMem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("LureMem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Mem", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatIntNoAggItem(etime.Train, etime.Run, "FirstPerfect")

	// ss.Logs.AddCopyFromFloatItems(etime.Train, etime.Epoch, etime.Test, etime.Epoch, "Tst", "PhaseDiff", "UnitErr", "PctCor", "PctErr", "TrgOnWasOffAll", "TrgOnWasOffCmp", "TrgOffWasOn", "Mem")
	ss.AddLogItems()

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	layers := ss.Net.LayersByType(leabra.SuperLayer, leabra.CTLayer, leabra.TargetLayer)
	leabra.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	leabra.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	// leabra.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "ActM", etime.Test, etime.Trial, "TargetLayer")
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "TargetLayer")

	ss.Logs.PlotItems("ABMem", "ACMem", "LureMem")

	// ss.Logs.PlotItems("TrgOnWasOffAll", "TrgOnWasOffCmp", "ABMem", "ACMem", "TstTrgOnWasOffAll", "TstTrgOnWasOffCmp", "TstMem", "TstABMem", "TstACMem")

	ss.Logs.CreateTables()
	ss.Logs.SetMeta(etime.Train, etime.Run, "TrgOnWasOffAll:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TrgOnWasOffCmp:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "ABMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "ACMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "LureMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstTrgOnWasOffAll:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstTrgOnWasOffCmp:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstABMem:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstACMem:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Run, "TstLureMem:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Run, "Type", "Bar")
	ss.Logs.SetMeta(etime.Train, etime.Epoch, "ABMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Epoch, "ACMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Epoch, "LureMem:On", "-")
	ss.Logs.SetMeta(etime.Train, etime.Epoch, "Mem:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Epoch, "TrgOnWasOffAll:On", "+")
	ss.Logs.SetMeta(etime.Train, etime.Epoch, "TrgOffWasOn:On", "+")
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	if mode != etime.Analyze {
		ctx.Mode = mode // Also set specifically in a Loop callback.
	}
	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
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

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Hippocampus"
	ss.GUI.MakeBody(ss, "hip", title, `runs a hippocampus model on the AB-AC paired associate learning task. See <a href="https://github.com/compcogneuro/sims/blob/master/ch7/hip/README.md">README.md on GitHub</a>.</p>`, readme)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.Raster.Max = 100
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	// nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1, 2.75)
	// nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.AddPlots(title, &ss.Logs)

	stnm := "RunStats"
	dt := ss.Logs.MiscTable(stnm)
	bcp, _ := ss.GUI.Tabs.NewTab(stnm + " Plot")
	plt := plotcore.NewSubPlot(bcp)
	ss.GUI.Plots[etime.ScopeKey(stnm)] = plt
	plt.Options.Title = "Run Stats"
	plt.Options.XAxis = "RunName"
	plt.SetTable(dt)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

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
			core.TheApp.OpenURL("https://github.com/compcogneuro/sims/blob/master/ch7/hip/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}
