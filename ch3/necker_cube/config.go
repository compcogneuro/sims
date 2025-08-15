// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package neckercube

import "github.com/emer/emergent/v2/egui"

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

	// Script is an interpreted script that is run to set parameters in Layer and Path
	// sheets, by default using the "Script" set name.
	Script string `new-window:"+" width:"100"`

	// Sheet is the extra params sheet name(s) to use (space separated
	// if multiple). Must be valid name as listed in compiled-in params
	// or loaded params.
	Sheet string

	// Tag is an extra tag to add to file names and logs saved from this run.
	Tag string

	// Note is additional info to describe the run params etc,
	// like a git commit message for the run.
	Note string
}

// RunConfig has config parameters related to running the sim.
type RunConfig struct {

	// Run is the _starting_ run number, which determines the random seed.
	// Runs counts up from there. Can do all runs in parallel by launching
	// separate jobs with each starting Run, Runs = 1.
	Run int `default:"0" flag:"run"`

	// Runs is the total number of runs to do when running Train, starting from Run.
	Runs int `default:"5" min:"1"`

	// Epochs is the total number of epochs per run.
	// Epochs int `default:"100"`

	// NZero is how many perfect, zero-error epochs before stopping a Run.
	// NZero int `default:"5"`

	// TestInterval is how often (in epochs) to run through all the test patterns,
	// in terms of training epochs. Can use 0 or -1 for no testing.
	// TestInterval int `default:"5"`
}

// LogConfig has config parameters related to logging data.
type LogConfig struct {
	// Test has the list of Test mode levels to save log files for.
	Test []string `nest:"+"`
}

// Config has the overall Sim configuration options.
type Config struct {
	egui.BaseConfig

	// Params has parameter related configuration options.
	Params ParamConfig `display:"add-fields"`

	// Run has sim running related configuration options.
	Run RunConfig `display:"add-fields"`

	// Log has data logging related configuration options.
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) Defaults() {
	cfg.Name = "CatsAndDogs"
	cfg.Title = "Cats and Dogs"
	cfg.URL = "https://github.com/CompCogNeuro/sims/blob/main/ch4/associator/README.md"
	cfg.Doc = "Explores a simple **semantic network** intended to represent a (very small) set of relationships among different features used to represent a set of entities in the world.  In our case, we represent some features of cats and dogs: their color, size, favorite food, and favorite toy."
}
